#pragma once

#include "common.cuh"
#include "../../include/ggml-turbo-kv.h"

// CUDA dispatch entry points for TURBO_KV_4B. Algorithm mirrors the Vulkan
// shaders in ggml/src/ggml-vulkan/vulkan-shaders/{turbo_kv_4b_rht.glsl,
// cpy_f32_turbo_kv_4b.comp, dequant_turbo_kv_4b.comp, set_rows_turbo_kv_4b.comp}.
// One CUDA block per 128-element TURBO_KV_4B block, 128 threads each, one
// element per lane. Butterfly stages 1..16 use __shfl_xor_sync within a warp;
// stages 32 and 64 use shared memory + __syncthreads.

void ggml_cuda_cpy_f32_turbo_kv_4b(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream);

void ggml_cuda_cpy_turbo_kv_4b_f32(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream);

void ggml_cuda_op_set_rows_turbo_kv_4b(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
