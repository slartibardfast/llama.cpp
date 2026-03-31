#pragma once

#include "ggml.h"
#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <memory>

struct ggml_cgraph;
struct vk_pipeline_struct;

namespace ggml_vk_jit {

using bda_fn = std::function<uint64_t(const ggml_tensor*)>;

// ===== GPU Interpreter (Phase 17c — Dolphin-style) =====

// Op types for the interpreter SSBO program
enum OpType : uint32_t {
    OP_ADD = 0, OP_MUL = 1, OP_SILU = 2, OP_SIGMOID = 3,
    OP_SCALE = 4, OP_CPY = 5, OP_SWIGLU_SPLIT = 6,
    OP_RMS_NORM = 7, OP_L2_NORM = 8,
};

// SSBO program entry — must match GLSL struct layout (std430, 40 bytes)
struct alignas(8) JitOp {
    uint32_t type;
    uint32_t ne;
    uint64_t src0;
    uint64_t src1;
    uint64_t dst;
    float    param;
    uint32_t _pad;
};
static_assert(sizeof(JitOp) == 40, "JitOp must be 40 bytes for std430 layout");

// Segment: a group of consecutive ops between hardware barriers
struct Segment {
    int start_node;         // graph node index (inclusive)
    int end_node;           // graph node index (exclusive)
    bool interpreter_eligible;  // true = use interpreter, false = standard dispatch
    bool needs_single_wg;       // true = reduction present, dispatch 1 WG
    std::vector<JitOp> program; // filled when interpreter_eligible
    uint32_t max_ne;            // largest element count (for WG calculation)
};

// Walk graph, identify segments and build interpreter programs.
std::vector<Segment> build_segments(
    const ggml_cgraph * cgraph,
    bda_fn get_bda);

// Get the interpreter GLSL source (compiled once, reused for all tokens).
const std::string & get_interpreter_glsl();

// ===== Shared utilities =====

std::vector<uint32_t> compile_glsl(
    const std::string & source,
    const std::string & name = "jit_interp");

std::string get_cache_path(const std::string & signature);
bool load_spirv_cache(const std::string & path, std::vector<uint32_t> & spirv);
bool save_spirv_cache(const std::string & path, const std::vector<uint32_t> & spirv);

} // namespace ggml_vk_jit
