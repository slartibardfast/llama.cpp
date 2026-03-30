#include "ggml-vk-jit.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <sstream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <cstring>

#ifdef GGML_VULKAN_JIT_SHADERC
#include <shaderc/shaderc.hpp>
#endif

// FNV-1a hash for cache keys (no external dependency)

namespace ggml_vk_jit {

// ========== GLSL Emitter Helpers ==========

static void emit_header(std::ostringstream & ss, const JitConfig & cfg) {
    ss << "#version 450\n"
       << "#extension GL_EXT_buffer_reference : require\n"
       << "#extension GL_EXT_shader_explicit_arithmetic_types : require\n"
       << "#extension GL_EXT_shader_16bit_storage : require\n"
       << "#extension GL_EXT_control_flow_attributes : enable\n"
       << "#extension GL_KHR_shader_subgroup_basic : require\n"
       << "#extension GL_KHR_shader_subgroup_arithmetic : require\n"
       << "\n"
       << "// BDA buffer reference types\n"
       << "layout(buffer_reference, std430, buffer_reference_align = 4)  readonly buffer FBuf  { float d[]; };\n"
       << "layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer FV4   { vec4  d[]; };\n"
       << "layout(buffer_reference, std430, buffer_reference_align = 4)  writeonly buffer FOut { float d[]; };\n"
       << "layout(buffer_reference, std430, buffer_reference_align = 4)  buffer FRW  { float d[]; };\n"
       << "\n"
       << "// Q4_K weight block (144 bytes, 256 elements)\n"
       << "struct Q4K_B { f16vec2 dm; uint16_t scales[6]; uint16_t qs[64]; };\n"
       << "layout(buffer_reference, std430, buffer_reference_align = 4)  readonly buffer Q4KBuf { Q4K_B b[]; };\n"
       << "struct Q4K_P32 { f16vec2 dm; uint32_t scales[3]; uint32_t qs[32]; };\n"
       << "layout(buffer_reference, std430, buffer_reference_align = 4)  readonly buffer Q4KP32 { Q4K_P32 b[]; };\n"
       << "\n"
       << "// Barrier infrastructure\n"
       << "layout(binding = 0) coherent buffer BarrierBuf { uint counters[2]; };\n"
       << "layout(push_constant) uniform PC { uint num_wgs; };\n"
       << "layout(local_size_x = " << cfg.workgroup_size << ", local_size_y = 1, local_size_z = 1) in;\n"
       << "\n"
       << "shared uint local_sense;\n"
       << "\n"
       << "void gbarrier() {\n"
       << "    memoryBarrierBuffer();\n"
       << "    if (gl_LocalInvocationID.x == 0) {\n"
       << "        uint s = local_sense;\n"
       << "        uint arrived = atomicAdd(counters[0], 1u) + 1u;\n"
       << "        if (arrived == num_wgs) {\n"
       << "            counters[0] = 0u;\n"
       << "            atomicExchange(counters[1], 1u - s);\n"
       << "        } else {\n"
       << "            while (atomicOr(counters[1], 0u) == s) {}\n"
       << "        }\n"
       << "        local_sense = 1u - s;\n"
       << "    }\n"
       << "    barrier();\n"
       << "    memoryBarrierBuffer();\n"
       << "}\n\n";
}

static std::string addr_lit(uint64_t addr) {
    std::ostringstream ss;
    ss << "0x" << std::hex << addr << "ul";
    return ss.str();
}

static void emit_elementwise(std::ostringstream & ss, const char * label,
                              const char * op_expr,
                              uint64_t src0, uint64_t dst, uint32_t ne,
                              uint64_t src1 = 0) {
    ss << "    // " << label << " ne=" << ne << "\n"
       << "    {\n"
       << "        FBuf s0 = FBuf(" << addr_lit(src0) << ");\n";
    if (src1) {
        ss << "        FBuf s1 = FBuf(" << addr_lit(src1) << ");\n";
    }
    ss << "        FOut dst = FOut(" << addr_lit(dst) << ");\n"
       << "        for (uint i = start; i < " << ne << "u; i += stride)\n"
       << "            " << op_expr << ";\n"
       << "    }\n"
       << "    gbarrier();\n\n";
}

// ========== Op Emitters ==========

static bool emit_op(std::ostringstream & ss, const ggml_tensor * node, bda_fn get_bda) {
    uint64_t dst_addr  = get_bda(node);
    uint64_t src0_addr = node->src[0] ? get_bda(node->src[0]) : 0;
    uint64_t src1_addr = node->src[1] ? get_bda(node->src[1]) : 0;
    uint32_t ne = (uint32_t)ggml_nelements(node);

    if (dst_addr == 0 || src0_addr == 0) return false;

    switch (node->op) {
    case GGML_OP_ADD:
        if (!src1_addr) return false;
        emit_elementwise(ss, "ADD", "dst.d[i] = s0.d[i] + s1.d[i]",
                          src0_addr, dst_addr, ne, src1_addr);
        return true;

    case GGML_OP_MUL:
        if (!src1_addr) return false;
        emit_elementwise(ss, "MUL", "dst.d[i] = s0.d[i] * s1.d[i]",
                          src0_addr, dst_addr, ne, src1_addr);
        return true;

    case GGML_OP_SCALE: {
        float scale;
        memcpy(&scale, node->op_params, sizeof(float));
        std::ostringstream expr;
        expr << "dst.d[i] = s0.d[i] * " << std::setprecision(9) << scale << "f";
        emit_elementwise(ss, "SCALE", expr.str().c_str(),
                          src0_addr, dst_addr, ne);
        return true;
    }

    case GGML_OP_CPY:
    case GGML_OP_CONT:
        emit_elementwise(ss, "CPY", "dst.d[i] = s0.d[i]",
                          src0_addr, dst_addr, ne);
        return true;

    case GGML_OP_UNARY: {
        auto uop = ggml_get_unary_op(node);
        if (uop == GGML_UNARY_OP_SILU) {
            emit_elementwise(ss, "SILU",
                "{ float x = s0.d[i]; dst.d[i] = x / (1.0 + exp(-x)); }",
                src0_addr, dst_addr, ne);
            return true;
        }
        if (uop == GGML_UNARY_OP_SIGMOID) {
            emit_elementwise(ss, "SIGMOID",
                "dst.d[i] = 1.0 / (1.0 + exp(-s0.d[i]))",
                src0_addr, dst_addr, ne);
            return true;
        }
        return false;
    }

    case GGML_OP_MUL_MAT: {
        // Only handle single-column mat-vec (n=1) for now
        if (node->ne[1] != 1) return false;
        // Only handle f32 activation
        if (!node->src[1] || node->src[1]->type != GGML_TYPE_F32) return false;

        uint32_t M = (uint32_t)node->ne[0];  // output rows
        uint32_t K = (uint32_t)node->src[1]->ne[0];  // input columns

        if (node->src[0]->type == GGML_TYPE_Q4_K) {
            uint32_t nblocks = K / 256;
            ss << "    // MUL_MAT_VEC q4_K M=" << M << " K=" << K << "\n"
               << "    {\n"
               << "        Q4KBuf A = Q4KBuf(" << addr_lit(src0_addr) << ");\n"
               << "        Q4KP32 A32 = Q4KP32(" << addr_lit(src0_addr) << ");\n"
               << "        FV4 Bv = FV4(" << addr_lit(src1_addr) << ");\n"
               << "        FOut D = FOut(" << addr_lit(dst_addr) << ");\n"
               << "        for (uint row = gl_WorkGroupID.x; row < " << M << "u; row += num_wgs) {\n"
               << "            float acc = 0.0;\n"
               << "            uint ib0 = row * " << nblocks << "u;\n"
               << "            for (uint blk = gl_LocalInvocationID.x; blk < " << nblocks << "u; blk += " << 64 << ") {\n"
               << "                vec2 dm = vec2(A.b[ib0 + blk].dm);\n"
               << "                // Simplified q4_K dot: iterate over 256 elements per block\n"
               << "                for (uint j = 0; j < 256u; j += 4u) {\n"
               << "                    uint qi = j / 2u;\n"
               << "                    uint qs_idx = qi / 2u;\n"
               << "                    uint qs_val = uint(A.b[ib0 + blk].qs[qs_idx]);\n"
               << "                    float q0 = float((qs_val >> 0) & 0xFu);\n"
               << "                    float q1 = float((qs_val >> 4) & 0xFu);\n"
               << "                    float q2 = float((qs_val >> 8) & 0xFu);\n"
               << "                    float q3 = float((qs_val >> 12) & 0xFu);\n"
               << "                    uint b_idx = blk * 256u + j;\n"
               << "                    vec4 bv = Bv.d[b_idx / 4u];\n"
               << "                    acc += dm.x * (bv.x * q0 + bv.y * q1 + bv.z * q2 + bv.w * q3);\n"
               << "                }\n"
               << "            }\n"
               << "            acc = subgroupAdd(acc);\n"
               << "            if (gl_SubgroupInvocationID == 0u) D.d[row] = acc;\n"
               << "        }\n"
               << "    }\n"
               << "    gbarrier();\n\n";
            return true;
        }

        // For other quant types, return false (handled by standard dispatch)
        return false;
    }

    default:
        return false;
    }
}

// ========== Public API ==========

std::string generate_shader(const ggml_cgraph * cgraph, bda_fn get_bda, const JitConfig & config) {
    std::ostringstream ss;

    emit_header(ss, config);

    // Main function
    ss << "void main() {\n"
       << "    if (gl_LocalInvocationID.x == 0) local_sense = 0u;\n"
       << "    barrier();\n"
       << "    const uint start = gl_WorkGroupID.x * " << config.workgroup_size << "u + gl_LocalInvocationID.x;\n"
       << "    const uint stride = num_wgs * " << config.workgroup_size << "u;\n\n";

    uint32_t emitted = 0;
    uint32_t skipped = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];

        // Skip empty/noop nodes
        if (ggml_is_empty(node)) { skipped++; continue; }
        if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE ||
            node->op == GGML_OP_PERMUTE) { skipped++; continue; }

        if (!emit_op(ss, node, get_bda)) {
            // Unsupported op — can't JIT the full graph
            fprintf(stderr, "ggml_vk_jit: unsupported op %s at node %d, aborting JIT\n",
                    ggml_op_name(node->op), i);
            return "";
        }
        emitted++;
    }

    ss << "}\n";

    fprintf(stderr, "ggml_vk_jit: generated %u ops (%u skipped) → %zu bytes GLSL\n",
            emitted, skipped, ss.str().size());
    return ss.str();
}

std::string compute_signature(const ggml_cgraph * cgraph, bda_fn get_bda) {
    // FNV-1a 64-bit hash
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&](const void * data, size_t len) {
        const uint8_t * p = (const uint8_t *)data;
        for (size_t i = 0; i < len; i++) {
            h ^= p[i];
            h *= 0x100000001b3ULL;
        }
    };

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        uint32_t op = node->op;
        mix(&op, sizeof(op));
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            mix(&node->ne[d], sizeof(node->ne[d]));
        }
        uint64_t bda = get_bda(node);
        mix(&bda, sizeof(bda));
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            uint64_t src_bda = node->src[s] ? get_bda(node->src[s]) : 0;
            mix(&src_bda, sizeof(src_bda));
        }
    }

    std::ostringstream ss;
    ss << std::hex << std::setw(16) << std::setfill('0') << h;
    return ss.str();
}

#ifdef GGML_VULKAN_JIT_SHADERC
std::vector<uint32_t> compile_glsl(const std::string & source, const std::string & name) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    options.SetOptimizationLevel(shaderc_optimization_level_performance);

    auto result = compiler.CompileGlslToSpv(source, shaderc_compute_shader, name.c_str(), options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        fprintf(stderr, "ggml_vk_jit: shader compilation failed:\n%s\n", result.GetErrorMessage().c_str());
        return {};
    }

    fprintf(stderr, "ggml_vk_jit: compiled %s → %zu SPIR-V words (%zu warnings)\n",
            name.c_str(), std::distance(result.begin(), result.end()), result.GetNumWarnings());

    return { result.begin(), result.end() };
}
#else
std::vector<uint32_t> compile_glsl(const std::string &, const std::string &) {
    fprintf(stderr, "ggml_vk_jit: shaderc not available, JIT disabled\n");
    return {};
}
#endif

// ========== Cache Management ==========

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
    fprintf(stderr, "ggml_vk_jit: loaded cached SPIR-V from %s (%zu words)\n", path.c_str(), spirv.size());
    return true;
}

bool save_spirv_cache(const std::string & path, const std::vector<uint32_t> & spirv) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(spirv.data()), spirv.size() * 4);
    fprintf(stderr, "ggml_vk_jit: saved SPIR-V cache to %s (%zu words)\n", path.c_str(), spirv.size());
    return true;
}

} // namespace ggml_vk_jit
