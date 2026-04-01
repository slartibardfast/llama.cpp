#include "ggml-vk-jit.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <filesystem>
#include <fstream>
#include <cstring>

#ifdef GGML_VULKAN_JIT_SHADERC
#include <shaderc/shaderc.hpp>
#endif

namespace ggml_vk_jit {

// ========== Interpreter GLSL (compiled once, reused forever) ==========

static const std::string interpreter_glsl = R"(#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(buffer_reference, std430, buffer_reference_align = 4) readonly  buffer FBuf { float d[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) writeonly buffer FOut { float d[]; };

const uint OP_ADD = 0u, OP_MUL = 1u, OP_SILU = 2u, OP_SIGMOID = 3u;
const uint OP_SCALE = 4u, OP_CPY = 5u, OP_SWIGLU_SPLIT = 6u;
const uint OP_RMS_NORM = 7u, OP_L2_NORM = 8u;

struct Op {
    uint type;
    uint ne;
    uint64_t src0;
    uint64_t src1;
    uint64_t dst;
    float param;
    uint _pad;
};

layout(binding = 0, std430) readonly buffer Program { uint num_ops; uint _pad0; Op ops[]; };
layout(push_constant) uniform PC { uint _dummy; };
layout(local_size_x = 256) in;

shared float smem[4];

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint stride = gl_NumWorkGroups.x * 256u;

    for (uint p = 0u; p < num_ops; p++) {
        Op op = ops[p];

        if (op.type <= OP_SWIGLU_SPLIT) {
            for (uint i = gid; i < op.ne; i += stride) {
                float v = FBuf(op.src0).d[i];
                switch (op.type) {
                case OP_ADD:          v = v + FBuf(op.src1).d[i]; break;
                case OP_MUL:          v = v * FBuf(op.src1).d[i]; break;
                case OP_SILU:         v = v / (1.0 + exp(-v)); break;
                case OP_SIGMOID:      v = 1.0 / (1.0 + exp(-v)); break;
                case OP_SCALE:        v = v * op.param; break;
                case OP_CPY:          break;
                case OP_SWIGLU_SPLIT: { float g = v; v = (g / (1.0 + exp(-g))) * FBuf(op.src1).d[i]; break; }
                }
                FOut(op.dst).d[i] = v;
            }
        } else if (op.type == OP_RMS_NORM || op.type == OP_L2_NORM) {
            float sq = 0.0;
            for (uint i = gl_LocalInvocationID.x; i < op.ne; i += 256u) {
                float v = FBuf(op.src0).d[i]; sq += v * v;
            }
            sq = subgroupAdd(sq);
            if (gl_SubgroupInvocationID == 0u) smem[gl_SubgroupID] = sq;
            barrier();
            if (gl_LocalInvocationID.x == 0u)
                smem[0] = smem[0] + smem[1] + smem[2] + smem[3];
            barrier();
            float sc;
            if (op.type == OP_RMS_NORM)
                sc = inversesqrt(smem[0] / float(op.ne) + op.param);
            else
                sc = 1.0 / max(sqrt(smem[0]), op.param);
            for (uint i = gl_LocalInvocationID.x; i < op.ne; i += 256u)
                FOut(op.dst).d[i] = FBuf(op.src0).d[i] * sc;
        }

        memoryBarrier();
        barrier();
    }
}
)";

const std::string & get_interpreter_glsl() { return interpreter_glsl; }

// ========== Op Classification ==========

bool is_interpreter_op(const ggml_tensor * t) {
    // Interpreter only handles contiguous f32 tensors
    if (t->type != GGML_TYPE_F32) return false;
    if (t->src[0] && t->src[0]->type != GGML_TYPE_F32) return false;

    switch (t->op) {
    case GGML_OP_ADD:
        // Element-wise only (no broadcasting)
        if (!t->src[1] || t->src[1]->type != GGML_TYPE_F32) return false;
        return ggml_nelements(t->src[0]) == ggml_nelements(t)
            && ggml_nelements(t->src[1]) == ggml_nelements(t);
    case GGML_OP_MUL:
        // Element-wise only (no broadcasting)
        if (!t->src[1] || t->src[1]->type != GGML_TYPE_F32) return false;
        return ggml_nelements(t->src[0]) == ggml_nelements(t)
            && ggml_nelements(t->src[1]) == ggml_nelements(t);
    case GGML_OP_SCALE:
        return true;
    case GGML_OP_RMS_NORM:
    case GGML_OP_L2_NORM:
        // Single-row only — multi-row needs per-row reduction
        return t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1;
    case GGML_OP_UNARY: {
        auto uop = ggml_get_unary_op(t);
        return uop == GGML_UNARY_OP_SILU || uop == GGML_UNARY_OP_SIGMOID;
    }
    case GGML_OP_GLU:
        if (ggml_get_glu_op(t) != GGML_GLU_OP_SWIGLU || !t->src[1]) return false;
        return t->src[1]->type == GGML_TYPE_F32;
    default:
        return false;
    }
}

bool build_jit_op(JitOp & out, const ggml_tensor * t, const bda_fn & get_bda) {
    out = {};
    out.dst = get_bda(t);
    out.src0 = t->src[0] ? get_bda(t->src[0]) : 0;
    out.src1 = t->src[1] ? get_bda(t->src[1]) : 0;
    out.ne = (uint32_t)ggml_nelements(t);
    if (!out.dst || !out.src0) return false;

    switch (t->op) {
    case GGML_OP_ADD:       out.type = OP_ADD; if (!out.src1) return false; break;
    case GGML_OP_MUL:       out.type = OP_MUL; if (!out.src1) return false; break;
    case GGML_OP_SCALE:     out.type = OP_SCALE; memcpy(&out.param, t->op_params, 4); break;
    case GGML_OP_CPY:
    case GGML_OP_CONT:      out.type = OP_CPY; break;
    case GGML_OP_RMS_NORM:  out.type = OP_RMS_NORM; memcpy(&out.param, t->op_params, 4); break;
    case GGML_OP_L2_NORM:   out.type = OP_L2_NORM; out.param = 1e-12f;
                             if (t->op_params[0]) memcpy(&out.param, t->op_params, 4); break;
    case GGML_OP_UNARY:
        if (ggml_get_unary_op(t) == GGML_UNARY_OP_SILU) out.type = OP_SILU;
        else if (ggml_get_unary_op(t) == GGML_UNARY_OP_SIGMOID) out.type = OP_SIGMOID;
        else return false;
        break;
    case GGML_OP_GLU:
        if (ggml_get_glu_op(t) != GGML_GLU_OP_SWIGLU || !t->src[1]) return false;
        out.type = OP_SWIGLU_SPLIT;
        if (!out.src1) return false;
        break;
    default:
        return false;
    }
    return true;
}

// ========== SPIR-V Compilation & Cache ==========

#ifdef GGML_VULKAN_JIT_SHADERC
std::vector<uint32_t> compile_glsl(const std::string & source, const std::string & name) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
    auto result = compiler.CompileGlslToSpv(source, shaderc_compute_shader, name.c_str(), options);
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
