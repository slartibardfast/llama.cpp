#pragma once

#include "ggml.h"
#include <string>
#include <vector>
#include <cstdint>
#include <functional>

// Phase 17b: JIT shader generator for single-dispatch inference.
// Walks the ggml compute graph and emits a specialized GLSL shader
// with hardcoded BDA addresses, baked dimensions, and straight-line
// op sequence. Compiled at runtime via shaderc.

struct ggml_cgraph;

namespace ggml_vk_jit {

// Callback to get BDA address for a tensor
using bda_fn = std::function<uint64_t(const ggml_tensor*)>;

struct JitConfig {
    uint32_t num_workgroups = 5;  // 1 per CU on Polaris 12
    uint32_t workgroup_size = 64; // 1 wavefront on GCN3
};

// Generate GLSL source for the entire compute graph.
// Returns empty string if any op is unsupported.
std::string generate_shader(
    const ggml_cgraph * cgraph,
    bda_fn get_bda,
    const JitConfig & config = {});

// Compute a signature for cache lookup.
// Hash of (op types + dimensions + BDA addresses).
std::string compute_signature(
    const ggml_cgraph * cgraph,
    bda_fn get_bda);

// Compile GLSL source to SPIR-V binary via shaderc.
// Returns empty vector on failure (error logged to stderr).
std::vector<uint32_t> compile_glsl(
    const std::string & source,
    const std::string & name = "jit_uber");

// Cache management
std::string get_cache_path(const std::string & signature);
bool load_spirv_cache(const std::string & path, std::vector<uint32_t> & spirv);
bool save_spirv_cache(const std::string & path, const std::vector<uint32_t> & spirv);

} // namespace ggml_vk_jit
