#include "ggml-vk-jit.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <sstream>
#include <set>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <cstring>

#ifdef GGML_VULKAN_JIT_SHADERC
#include <shaderc/shaderc.hpp>
#endif

namespace ggml_vk_jit {

// ========== Helpers ==========

static bool is_noop(const ggml_tensor * t) {
    return !t || ggml_is_empty(t) ||
           t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW ||
           t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_PERMUTE;
}

static bool is_elementwise(const ggml_tensor * t) {
    switch (t->op) {
    case GGML_OP_ADD:
    case GGML_OP_MUL:
    case GGML_OP_SCALE:
    case GGML_OP_CPY:
    case GGML_OP_CONT:
        return true;
    case GGML_OP_UNARY: {
        auto uop = ggml_get_unary_op(t);
        return uop == GGML_UNARY_OP_SILU || uop == GGML_UNARY_OP_SIGMOID;
    }
    case GGML_OP_GLU:
        return ggml_get_glu_op(t) == GGML_GLU_OP_SWIGLU;
    default:
        return false;
    }
}

static bool is_reduction(const ggml_tensor * t) {
    return t->op == GGML_OP_RMS_NORM || t->op == GGML_OP_L2_NORM;
}

static std::string A(uint64_t addr) {
    std::ostringstream ss;
    ss << "0x" << std::hex << addr << "ul";
    return ss.str();
}

// ========== Fusion Group Builder ==========

// Simplified dependency check: does tensor 'a' overlap in memory with any tensor in 'list'?
// Uses raw data pointers for overlap detection (avoids ggml_backend_buffer internals).
static bool overlaps_any(const ggml_tensor * a, const std::vector<const ggml_tensor *> & list) {
    if (!a || !a->data || list.empty()) return false;
    auto a_begin = (uintptr_t)a->data;
    auto a_end = a_begin + ggml_nbytes(a);
    for (const auto * other : list) {
        if (!other || !other->data) continue;
        auto o_begin = (uintptr_t)other->data;
        auto o_end = o_begin + ggml_nbytes(other);
        if (a_begin < o_end && o_begin < a_end)
            return true;
    }
    return false;
}

std::vector<FusionGroup> build_fusion_groups(const ggml_cgraph * cgraph, bda_fn get_bda) {
    std::vector<FusionGroup> groups;
    std::vector<const ggml_tensor *> unsynced_written;
    std::vector<const ggml_tensor *> unsynced_read;

    int group_start = -1;
    bool in_group = false;

    auto finish_group = [&](int end_node) {
        if (!in_group) return;
        FusionGroup g;
        g.start_node = group_start;
        g.end_node = end_node;

        // Classify: scan all compute nodes in this group
        bool all_ew = true, starts_reduction = false;
        int compute_count = 0;
        for (int i = g.start_node; i < g.end_node; i++) {
            const ggml_tensor * n = cgraph->nodes[i];
            if (is_noop(n)) continue;
            compute_count++;
            if (!is_elementwise(n)) all_ew = false;
            if (compute_count == 1 && is_reduction(n)) starts_reduction = true;
        }

        if (compute_count == 0) {
            in_group = false;
            return;
        }

        // Check if we can JIT this group
        bool can_jit = false;
        if (all_ew && compute_count >= 2) {
            // Elementwise chain of 2+ ops — worth fusing
            g.type = GroupType::ELEMENTWISE_CHAIN;
            // Verify all ops have valid BDA
            can_jit = true;
            for (int i = g.start_node; i < g.end_node && can_jit; i++) {
                const ggml_tensor * n = cgraph->nodes[i];
                if (is_noop(n)) continue;
                if (!get_bda(n)) can_jit = false;
                if (n->src[0] && !get_bda(n->src[0])) can_jit = false;
                if (n->src[1] && n->op != GGML_OP_SCALE && !get_bda(n->src[1])) can_jit = false;
            }
        } else if (starts_reduction && compute_count >= 2) {
            // Reduction followed by elementwise tail
            g.type = GroupType::REDUCTION_CHAIN;
            // TODO: implement reduction+tail emitter
            can_jit = false;
        }

        if (!can_jit) {
            g.type = GroupType::USE_STANDARD;
        }

        groups.push_back(std::move(g));
        in_group = false;
    };

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (is_noop(node)) {
            if (!in_group) { group_start = i; in_group = true; }
            continue;
        }

        // Check if this node needs a barrier (reads from something that was written)
        bool need_sync = false;
        // Destination overlaps with unsynced reads or writes?
        if (overlaps_any(node, unsynced_read) || overlaps_any(node, unsynced_written))
            need_sync = true;
        // Sources overlap with unsynced writes?
        for (int s = 0; s < GGML_MAX_SRC && !need_sync; s++) {
            if (node->src[s] && overlaps_any(node->src[s], unsynced_written))
                need_sync = true;
        }

        if (need_sync) {
            // Barrier point — finish current group, start new one
            finish_group(i);
            unsynced_written.clear();
            unsynced_read.clear();
        }

        if (!in_group) { group_start = i; in_group = true; }

        // Track this node's reads and writes
        unsynced_written.push_back(node);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (node->src[s]) unsynced_read.push_back(node->src[s]);
        }
    }
    finish_group(cgraph->n_nodes);

    return groups;
}

// ========== Elementwise Chain GLSL Emitter ==========

std::string generate_group_shader(const ggml_cgraph * cgraph, const FusionGroup & group, bda_fn get_bda) {
    if (group.type != GroupType::ELEMENTWISE_CHAIN) return "";

    // Collect compute nodes
    struct OpInfo {
        const ggml_tensor * node;
        uint64_t dst, s0, s1;
    };
    std::vector<OpInfo> ops;
    for (int i = group.start_node; i < group.end_node; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (is_noop(n)) continue;
        OpInfo op;
        op.node = n;
        op.dst = get_bda(n);
        op.s0 = n->src[0] ? get_bda(n->src[0]) : 0;
        op.s1 = n->src[1] ? get_bda(n->src[1]) : 0;
        if (!op.dst) return "";
        ops.push_back(op);
    }
    if (ops.empty()) return "";

    // Use the LAST op's element count as the grid size
    uint32_t ne = (uint32_t)ggml_nelements(ops.back().node);

    std::ostringstream ss;
    ss << R"(#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require

layout(buffer_reference, std430, buffer_reference_align = 4) readonly  buffer FBuf { float d[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) writeonly buffer FOut { float d[]; };

layout(push_constant) uniform PC { uint ne; };
layout(local_size_x = 256) in;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= ne) return;

)";

    // Emit each op as inline code
    for (size_t idx = 0; idx < ops.size(); idx++) {
        auto & op = ops[idx];
        const ggml_tensor * n = op.node;
        std::string src0 = "FBuf(" + A(op.s0) + ").d[i]";
        std::string src1 = op.s1 ? "FBuf(" + A(op.s1) + ").d[i]" : "";
        std::string dst = (idx == ops.size() - 1)
            ? "FOut(" + A(op.dst) + ").d[i]"
            : "FBuf(" + A(op.dst) + ").d[i]";
        // For intermediate ops that write to a buffer that the next op reads,
        // use FOut for the write (writeonly). But actually we need read-write
        // for intermediates. Use a temp variable instead when possible.

        ss << "    // " << ggml_op_name(n->op) << "\n";

        switch (n->op) {
        case GGML_OP_ADD:
            ss << "    " << dst << " = " << src0 << " + " << src1 << ";\n";
            break;
        case GGML_OP_MUL:
            ss << "    " << dst << " = " << src0 << " * " << src1 << ";\n";
            break;
        case GGML_OP_SCALE: {
            float sc; memcpy(&sc, n->op_params, sizeof(float));
            ss << "    " << dst << " = " << src0 << " * " << std::setprecision(9) << sc << ";\n";
            break;
        }
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            ss << "    " << dst << " = " << src0 << ";\n";
            break;
        case GGML_OP_UNARY: {
            auto uop = ggml_get_unary_op(n);
            if (uop == GGML_UNARY_OP_SILU) {
                ss << "    { float _x = " << src0 << "; " << dst << " = _x / (1.0 + exp(-_x)); }\n";
            } else if (uop == GGML_UNARY_OP_SIGMOID) {
                ss << "    " << dst << " = 1.0 / (1.0 + exp(-" << src0 << "));\n";
            }
            break;
        }
        case GGML_OP_GLU: {
            // SWIGLU split variant (two inputs)
            if (n->src[1]) {
                ss << "    { float _g = " << src0 << "; " << dst << " = (_g / (1.0 + exp(-_g))) * " << src1 << "; }\n";
            } else {
                // Single-input SWIGLU — more complex, skip for now
                return "";
            }
            break;
        }
        default:
            return "";
        }
    }

    ss << "}\n";
    return ss.str();
}

// ========== Group Signature ==========

std::string compute_group_signature(const ggml_cgraph * cgraph, const FusionGroup & group, bda_fn get_bda) {
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&](const void * data, size_t len) {
        const uint8_t * p = (const uint8_t *)data;
        for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= 0x100000001b3ULL; }
    };
    for (int i = group.start_node; i < group.end_node; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (is_noop(node)) continue;
        uint32_t op = node->op;
        mix(&op, sizeof(op));
        for (int d = 0; d < GGML_MAX_DIMS; d++) mix(&node->ne[d], sizeof(node->ne[d]));
        uint64_t bda = get_bda(node);
        mix(&bda, sizeof(bda));
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            uint64_t src_bda = node->src[s] ? get_bda(node->src[s]) : 0;
            mix(&src_bda, sizeof(src_bda));
        }
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << h;
    return out.str();
}

// ========== SPIR-V Compilation & Cache ==========

#ifdef GGML_VULKAN_JIT_SHADERC
std::vector<uint32_t> compile_glsl(const std::string & source, const std::string & name) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
    auto result = compiler.CompileGlslToSpv(source, shaderc_compute_shader, name.c_str(), options);
    // Dump for debug
    FILE * dbg = fopen("/tmp/jit_debug.glsl", "w");
    if (dbg) { fputs(source.c_str(), dbg); fclose(dbg); }
    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        fprintf(stderr, "ggml_vk_jit: compilation failed:\n%s\n", result.GetErrorMessage().c_str());
        return {};
    }
    return { result.begin(), result.end() };
}
#else
std::vector<uint32_t> compile_glsl(const std::string &, const std::string &) {
    fprintf(stderr, "ggml_vk_jit: shaderc not available\n");
    return {};
}
#endif

std::string get_cache_path(const std::string & signature) {
    std::string dir = std::string(getenv("HOME") ? getenv("HOME") : "/tmp") + "/.cache/llama-vulkan-jit";
    std::filesystem::create_directories(dir);
    return dir + "/" + signature + ".spv";
}

bool load_spirv_cache(const std::string & path, std::vector<uint32_t> & spirv) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    if (size == 0 || size % 4 != 0) return false;
    f.seekg(0);
    spirv.resize(size / 4);
    f.read(reinterpret_cast<char*>(spirv.data()), size);
    return true;
}

bool save_spirv_cache(const std::string & path, const std::vector<uint32_t> & spirv) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(spirv.data()), spirv.size() * 4);
    return true;
}

} // namespace ggml_vk_jit
