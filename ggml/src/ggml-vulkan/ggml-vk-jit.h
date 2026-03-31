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

// ===== Fusion Group Architecture (Phase 17c) =====

enum class GroupType {
    ELEMENTWISE_CHAIN,  // ADD, MUL, SILU, SIGMOID, SCALE, CPY — JIT fused
    REDUCTION_CHAIN,    // RMS_NORM→MUL, L2_NORM→MUL — JIT fused
    USE_STANDARD,       // matmul, GDN, flash_attn, etc. — standard dispatch
};

struct FusionGroup {
    int start_node;     // first graph node index (inclusive)
    int end_node;       // last graph node index (exclusive)
    GroupType type;
    std::shared_ptr<vk_pipeline_struct> pipeline;  // JIT pipeline (null = not yet compiled)
    std::vector<uint32_t> spirv;                   // cached SPIR-V
    std::string signature;                         // cache key
};

// Walk the graph and identify fusion groups.
// Uses the same dependency logic as standard dispatch to find barrier points.
std::vector<FusionGroup> build_fusion_groups(
    const ggml_cgraph * cgraph,
    bda_fn get_bda);

// Generate GLSL for a fusion group (elementwise chain or reduction+tail).
// Returns empty string if the group can't be JIT-compiled.
std::string generate_group_shader(
    const ggml_cgraph * cgraph,
    const FusionGroup & group,
    bda_fn get_bda);

// Compute signature for a single fusion group.
std::string compute_group_signature(
    const ggml_cgraph * cgraph,
    const FusionGroup & group,
    bda_fn get_bda);

// ===== Shared utilities =====

std::vector<uint32_t> compile_glsl(
    const std::string & source,
    const std::string & name = "jit_fused");

std::string get_cache_path(const std::string & signature);
bool load_spirv_cache(const std::string & path, std::vector<uint32_t> & spirv);
bool save_spirv_cache(const std::string & path, const std::vector<uint32_t> & spirv);

} // namespace ggml_vk_jit
