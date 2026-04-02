#pragma once

#include "ggml.h"
#include <string>
#include <vector>
#include <cstdint>
#include <functional>

namespace ggml_vk_jit {

using bda_fn = std::function<uint64_t(const ggml_tensor*)>;

// Op types for the interpreter SSBO program
enum OpType : uint32_t {
    OP_ADD = 0, OP_MUL = 1, OP_SILU = 2, OP_SIGMOID = 3,
    OP_SCALE = 4, OP_CPY = 5, OP_SWIGLU_SPLIT = 6,
    OP_RMS_NORM = 7, OP_L2_NORM = 8,
    OP_FUSED_GATE_PREP = 9, OP_SSM_CONV = 10,
};

// SSBO program entry — must match GLSL struct layout (std430, 48 bytes)
struct alignas(8) JitOp {
    uint32_t type;    //  0
    uint32_t ne;      //  4  total elements
    uint64_t src0;    //  8
    uint64_t src1;    // 16
    uint64_t dst;     // 24
    float    param;   // 32
    uint32_t ne0;     // 36  row length / num_v_heads / nr
    uint64_t src2;    // 40  third source (fused_gate_prep: ssm_a, ssm_conv: unused)
};
static_assert(sizeof(JitOp) == 48, "JitOp must be 48 bytes for std430 layout");

// Check if a tensor op can be handled by the interpreter
bool is_interpreter_op(const ggml_tensor * t);

// Build a JitOp from a tensor node. Returns false if BDA addresses unavailable.
bool build_jit_op(JitOp & out, const ggml_tensor * t, const bda_fn & get_bda);

// Get the interpreter GLSL source (compiled once, reused for all tokens)
const std::string & get_interpreter_glsl();

// SPIR-V compilation and caching
std::vector<uint32_t> compile_glsl(const std::string & source, const std::string & name = "jit_interp");
std::string get_cache_path(const std::string & signature);
bool load_spirv_cache(const std::string & path, std::vector<uint32_t> & spirv);
bool save_spirv_cache(const std::string & path, const std::vector<uint32_t> & spirv);

} // namespace ggml_vk_jit
