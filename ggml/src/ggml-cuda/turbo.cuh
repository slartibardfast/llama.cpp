#pragma once

#include "common.cuh"
#include "../../include/ggml-turbo-kv.h"

// Fused RHT-space MUL_MAT for TURBO_2B / 3B / 4B / 5B weight types.
// Mirrors the Vulkan mul_mat_vec_turbo.comp algorithm. Per-device codebook
// buffer (4 * sizeof(float) for 2-bit through 5-bit) populated lazily on
// first use from the published Lloyd-Max constants in ggml-turbo-kv.c;
// can be overwritten at runtime to install imatrix-weighted centroids.

bool ggml_cuda_can_mul_mat_turbo(const ggml_tensor * src0);
void ggml_cuda_mul_mat_turbo(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst);

// Override the device-side codebook for a given bitrate. centroids must
// point to host memory of length 2^bits floats. Called by the imatrix
// loader path in llama.cpp; safe to call at any time.
void ggml_cuda_set_turbo_codebook(int device, int bits, const float * centroids);
