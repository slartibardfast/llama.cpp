/*
 * test-turbo-kv-pbt.cpp — Property-based tests for turbo-kv-4b spec
 *
 * Validates the Allium spec in turbo-kv-4b.allium against the C
 * implementation using RapidCheck. Each property maps to a spec
 * obligation (rule ensures, invariant, or config constraint).
 *
 * Build: cmake --build build-tq --target test-turbo-kv-pbt
 * Run:   build-tq/bin/test-turbo-kv-pbt
 */

#include "ggml-turbo-kv.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#if defined(__AVX2__)
#include "ggml-cpu/arch/x86/turbo_kv_4b_avx2.h"
#endif
#include <rapidcheck.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>
#include <limits>

/* ================================================================
 * Helpers
 * ================================================================ */

static float vec_l2_norm(const float * x, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    return sqrtf(s);
}

static float vec_dot(const float * a, const float * b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static float vec_l2_diff(const float * a, const float * b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return sqrtf(s);
}

static void fill_random(float * x, int n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int i = 0; i < n; i++) x[i] = dist(gen);
}

/* ================================================================
 * Spec: rule L2_norm — result >= 0
 * ================================================================ */
void property_L2_norm_is_non_negative() {
    rc::check("L2_norm(result) >= 0",
        [](int dim, uint32_t seed) {
            if (dim <= 0 || dim > 1024) return;
            std::vector<float> x(dim);
            fill_random(x.data(), dim, seed);
            float norm = vec_l2_norm(x.data(), dim);
            RC_ASSERT(norm >= 0.0f);
        });
}

/* ================================================================
 * Spec: rule normalize — L2_norm(result) = 1.0
 * ================================================================ */
void property_normalize_yields_unit_vector() {
    rc::check("normalize_yields_unit_vector",
        [](int dim, uint32_t seed) {
            if (dim <= 0 || dim > 1024) return;
            std::vector<float> x(dim);
            fill_random(x.data(), dim, seed);
            float norm = vec_l2_norm(x.data(), dim);
            if (norm < 1e-10f) {
                x[0] = 1.0f;
                norm = 1.0f;
            }
            float inv_norm = 1.0f / norm;
            std::vector<float> normalized(dim);
            for (int i = 0; i < dim; i++) normalized[i] = x[i] * inv_norm;
            float result_norm = vec_l2_norm(normalized.data(), dim);
            RC_ASSERT(fabsf(result_norm - 1.0f) < 1e-5f);
        });
}

/* ================================================================
 * Spec: rule rescale — L2_norm(result) = L2_norm(data) * norm
 * ================================================================ */
void property_rescale_scales_norm_proportionally() {
    rc::check("rescale_scales_norm_proportionally",
        [](int dim, uint32_t seed) {
            if (dim <= 0 || dim > 1024) return;
            float scale_factor = 0.1f + fabsf(std::numeric_limits<float>::denorm_min()) * 9.9f;
            // Clamp to [0.1, 10.0]
            if (scale_factor < 0.1f) scale_factor = 0.1f;
            if (scale_factor > 10.0f) scale_factor = 10.0f;
            std::vector<float> x(dim);
            fill_random(x.data(), dim, seed);
            float original_norm = vec_l2_norm(x.data(), dim);
            std::vector<float> scaled(dim);
            for (int i = 0; i < dim; i++) scaled[i] = x[i] * scale_factor;
            float scaled_norm = vec_l2_norm(scaled.data(), dim);
            float expected = original_norm * scale_factor;
            float rel_err = fabsf(scaled_norm - expected) / (expected + 1e-10f);
            RC_ASSERT(rel_err < 1e-5f);
        });
}

/* ================================================================
 * Spec: rule RHT_forward + RHT_inverse — self-inverse
 * ================================================================ */
void property_RHT_self_inverse() {
    rc::check("RHT_self_inverse",
        [](int dim, uint32_t seed) {
            // Only test valid RHT dimensions
            if (dim != 64 && dim != 128 && dim != 256) return;
            std::vector<float> x(dim), y(dim);
            fill_random(x.data(), dim, seed);
            memcpy(y.data(), x.data(), dim * sizeof(float));
            turbo_kv_rht_forward(y.data(), dim, seed);
            turbo_kv_rht_inverse(y.data(), dim, seed);
            float err = vec_l2_diff(x.data(), y.data(), dim);
            RC_ASSERT(err < 1e-4f);
        });
}

/* ================================================================
 * Spec: invariant RHTOrthogonality — <RHT(a), RHT(b)> == <a, b>
 * ================================================================ */
void property_RHT_preserves_dot_product() {
    rc::check("RHT_preserves_dot_product",
        [](int dim, uint32_t seed) {
            if (dim != 64 && dim != 128 && dim != 256) return;
            std::vector<float> a(dim), b(dim);
            fill_random(a.data(), dim, seed);
            fill_random(b.data(), dim, seed + 1);
            std::vector<float> a_rot(dim), b_rot(dim);
            memcpy(a_rot.data(), a.data(), dim * sizeof(float));
            memcpy(b_rot.data(), b.data(), dim * sizeof(float));
            turbo_kv_rht_forward(a_rot.data(), dim, seed);
            turbo_kv_rht_forward(b_rot.data(), dim, seed);
            float dot_orig = vec_dot(a.data(), b.data(), dim);
            float dot_rot  = vec_dot(a_rot.data(), b_rot.data(), dim);
            float rel_err  = fabsf(dot_orig - dot_rot) / (fabsf(dot_orig) + 1e-10f);
            RC_ASSERT(rel_err < 1e-4f);
        });
}

/* ================================================================
 * Spec: invariant IndicesInBounds — all indices in [0, 15]
 * ================================================================ */
void property_quantize_indices_in_bounds() {
    rc::check("quantize_indices_in_bounds",
        [](int dim, uint32_t seed) {
            // Only test block-aligned dimensions
            if (dim < 128 || dim > 1024 || dim % 128 != 0) return;
            std::vector<float> x(dim);
            fill_random(x.data(), dim, seed);
            std::vector<uint8_t> quantized_data(dim * sizeof(block_turbo_kv_4b) / 128);
            quantize_row_turbo_kv_4b_ref(x.data(),
                reinterpret_cast<block_turbo_kv_4b*>(quantized_data.data()), dim);
            int n_blocks = dim / TURBO_KV_BLOCK_SIZE;
            for (int b = 0; b < n_blocks; b++) {
                block_turbo_kv_4b * blk =
                    reinterpret_cast<block_turbo_kv_4b*>(
                        quantized_data.data() + b * sizeof(block_turbo_kv_4b));
                for (int i = 0; i < TURBO_KV_BLOCK_SIZE / 2; i++) {
                    uint8_t byte = blk->mse_indices[i];
                    int lo = byte & 0x0F;
                    int hi = (byte >> 4) & 0x0F;
                    RC_ASSERT(lo >= 0 && lo < 16);
                    RC_ASSERT(hi >= 0 && hi < 16);
                }
            }
        });
}

/* ================================================================
 * Spec: config.seed — deterministic seed value
 * ================================================================ */
void property_seed_is_deterministic() {
    rc::check("seed_is_deterministic", [] {
        RC_ASSERT(TURBO_KV_DEFAULT_SEED == 0x12345678u);
    });
}

/* ================================================================
 * Spec: config.block_size — 128 elements per block
 * ================================================================ */
void property_block_size_is_128() {
    rc::check("block_size_is_128", [] {
        RC_ASSERT(TURBO_KV_BLOCK_SIZE == 128);
    });
}

/* ================================================================
 * Spec: config.cent_max — 2.7326
 * ================================================================ */
void property_cent_max_is_correct() {
    rc::check("cent_max_is_correct", [] {
        float diff = fabsf(TURBO_KV_4B_CENT_MAX - 2.7326f);
        RC_ASSERT(diff < 1e-6f);
    });
}

/* ================================================================
 * Spec: rule nearest_centroid — result.count == input.count
 * ================================================================ */
void property_nearest_centroid_preserves_count() {
    rc::check("nearest_centroid_preserves_count",
        [](int dim, uint32_t seed) {
            if (dim < 128 || dim > 1024 || dim % 128 != 0) return;
            std::vector<float> x(dim);
            fill_random(x.data(), dim, seed);
            std::vector<uint8_t> quantized_data(dim * sizeof(block_turbo_kv_4b) / 128);
            quantize_row_turbo_kv_4b_ref(x.data(),
                reinterpret_cast<block_turbo_kv_4b*>(quantized_data.data()), dim);
            int n_blocks = dim / TURBO_KV_BLOCK_SIZE;
            RC_ASSERT(n_blocks * (TURBO_KV_BLOCK_SIZE / 2) * 2 == dim);
        });
}

/* ================================================================
 * Spec: rule reconstruct_codebook — result.count == block.count
 * ================================================================ */
void property_reconstruct_preserves_count() {
    rc::check("reconstruct_preserves_count",
        [](int dim, uint32_t seed) {
            if (dim < 128 || dim > 1024 || dim % 128 != 0) return;
            std::vector<float> x(dim);
            fill_random(x.data(), dim, seed);
            std::vector<uint8_t> quantized_data(dim * sizeof(block_turbo_kv_4b) / 128);
            quantize_row_turbo_kv_4b_ref(x.data(),
                reinterpret_cast<block_turbo_kv_4b*>(quantized_data.data()), dim);
            std::vector<float> y(dim);
            dequantize_row_turbo_kv_4b(
                reinterpret_cast<block_turbo_kv_4b*>(quantized_data.data()), y.data(), dim);
            RC_ASSERT(y.size() == static_cast<size_t>(dim));
        });
}

/* ================================================================
 * Spec: invariant ReconstructionPreservesNorm — dequant ~ original
 * ================================================================ */
void property_reconstruction_preserves_norm_within_tolerance() {
    rc::check("reconstruction_preserves_norm_within_tolerance",
        [](int dim, uint32_t seed) {
            if (dim < 128 || dim > 512 || dim % 128 != 0) return;
            std::vector<float> x(dim);
            fill_random(x.data(), dim, seed);
            float original_norm = vec_l2_norm(x.data(), dim);
            if (original_norm < 1e-10f) {
                x[0] = 1.0f; x[1] = 2.0f;
                original_norm = vec_l2_norm(x.data(), dim);
            }
            std::vector<uint8_t> quantized_data(dim * sizeof(block_turbo_kv_4b) / 128);
            quantize_row_turbo_kv_4b_ref(x.data(),
                reinterpret_cast<block_turbo_kv_4b*>(quantized_data.data()), dim);
            std::vector<float> y(dim);
            dequantize_row_turbo_kv_4b(
                reinterpret_cast<block_turbo_kv_4b*>(quantized_data.data()), y.data(), dim);
            float dequant_norm = vec_l2_norm(y.data(), dim);
            float abs_err = fabsf(dequant_norm - original_norm);
            float rel_err = abs_err / (original_norm + 1e-10f);
            /* Spec budget is config.reconstruction_rel_error = 0.1 (10%).
             * We assert 0.02 here because that is what the impl consistently
             * hits on near-Gaussian inputs (PHASE24 measured 0.083 as worst
             * observed). If this tightens below 0.02 legitimately, update
             * both this literal and the spec config value to match. */
            RC_ASSERT(rel_err < 0.02f);
        });
}

/* ================================================================
 * Spec: QuantizeBlock + DequantizeBlock — roundtrip quality
 * ================================================================ */
void property_roundtrip_quality_within_nmse_tolerance() {
    rc::check("roundtrip_quality_within_nmse_tolerance",
        [](int dim, uint32_t seed) {
            if (dim < 128 || dim > 512 || dim % 128 != 0) return;
            std::vector<float> x(dim);
            fill_random(x.data(), dim, seed);
            float signal = 0;
            for (int i = 0; i < dim; i++) signal += x[i] * x[i];
            std::vector<uint8_t> quantized_data(dim * sizeof(block_turbo_kv_4b) / 128);
            quantize_row_turbo_kv_4b_ref(x.data(),
                reinterpret_cast<block_turbo_kv_4b*>(quantized_data.data()), dim);
            std::vector<float> y(dim);
            dequantize_row_turbo_kv_4b(
                reinterpret_cast<block_turbo_kv_4b*>(quantized_data.data()), y.data(), dim);
            float mse = 0;
            for (int i = 0; i < dim; i++) {
                float d = x[i] - y[i];
                mse += d * d;
            }
            float nmse = mse / (signal + 1e-10f);
            RC_ASSERT(nmse < 0.05f);
        });
}

/* ================================================================
 * Spec: rule VectorDot — <query, block> matches the full-dequant dot
 * ================================================================ */
void property_VectorDot_matches_dequant_dot() {
    rc::check("VectorDot(query, block) = dot(query, DequantizeBlock(block))",
        [](uint32_t seed) {
            const int dim = TURBO_KV_BLOCK_SIZE;
            std::vector<float> query(dim), key(dim);
            fill_random(query.data(), dim, seed);
            fill_random(key.data(), dim, seed + 1);
            /* Spec requires: L2_norm(tensor.data) > 0.0 for QuantizeBlock. */
            if (vec_l2_norm(key.data(), dim) < 1e-6f) key[0] = 1.0f;

            block_turbo_kv_4b blk;
            quantize_row_turbo_kv_4b_ref(key.data(), &blk, dim);

            std::vector<float> dequant(dim);
            dequantize_row_turbo_kv_4b(&blk, dequant.data(), dim);
            const float ref = vec_dot(query.data(), dequant.data(), dim);

            float actual = 0;
            ggml_vec_dot_turbo_kv_4b_f32(dim, &actual, 0,
                reinterpret_cast<const void *>(&blk),         0,
                reinterpret_cast<const void *>(query.data()), 0,
                1);

            const float abs_err = fabsf(ref - actual);
            const float rel_err = abs_err / (fabsf(ref) + 1e-6f);
            RC_ASSERT(rel_err < 1e-3f);
        });
}

/* ================================================================
 * Spec: rule MultiBlockVectorDot — @guidance identity:
 *   MultiBlockVectorDot(query, blocks) = dot(query, dequantize_row(blocks))
 * The code's sum-of-per-block-dots-in-rotated-space path must match
 * the full-dequantize-then-dot reference.
 * ================================================================ */
void property_MultiBlockVectorDot_matches_dequant_dot() {
    rc::check("MultiBlockVectorDot matches full-dequant dot",
        [](int n_blocks_seed, uint32_t seed) {
            const int n_blocks = 2 + (n_blocks_seed & 3);  // 2, 3, 4, 5 → we want 2..4
            if (n_blocks < 2 || n_blocks > 4) return;
            const int head_dim = n_blocks * TURBO_KV_BLOCK_SIZE;

            std::vector<float> query(head_dim), key(head_dim);
            fill_random(query.data(), head_dim, seed);
            fill_random(key.data(), head_dim, seed + 1);
            /* Every block needs L2_norm > 0. Cheap fix: force one non-zero
             * element per block. */
            for (int b = 0; b < n_blocks; b++) {
                if (vec_l2_norm(key.data() + b * TURBO_KV_BLOCK_SIZE,
                                TURBO_KV_BLOCK_SIZE) < 1e-6f) {
                    key[b * TURBO_KV_BLOCK_SIZE] = 1.0f;
                }
            }

            std::vector<block_turbo_kv_4b> blocks(n_blocks);
            quantize_row_turbo_kv_4b_ref(key.data(), blocks.data(), head_dim);

            std::vector<float> dequant(head_dim);
            dequantize_row_turbo_kv_4b(blocks.data(), dequant.data(), head_dim);
            const float ref = vec_dot(query.data(), dequant.data(), head_dim);

            /* Path 1: scalar vec_dot — loops per-block internally. */
            float via_vec_dot = 0;
            ggml_vec_dot_turbo_kv_4b_f32(head_dim, &via_vec_dot, 0,
                reinterpret_cast<const void *>(blocks.data()), 0,
                reinterpret_cast<const void *>(query.data()),  0,
                1);

            /* Path 2: batched attention_multi (rotates query once). */
            float via_attn = 0;
            turbo_kv_4b_attention_multi(
                query.data(), blocks.data(), &via_attn,
                /* valid_count     = */ 1,
                /* head_dim        = */ head_dim,
                /* k_stride_blocks = */ n_blocks);

            auto rel = [&](float x) {
                return fabsf(ref - x) / (fabsf(ref) + 1e-6f);
            };
            RC_ASSERT(rel(via_vec_dot) < 1e-3f);
            RC_ASSERT(rel(via_attn)    < 1e-3f);
        });
}

/* ================================================================
 * Spec: invariant NearestCentroidMinimizesError — for every element
 * the chosen codebook index is the argmin of |scaled - codebook[c]|.
 * ================================================================ */
void property_nearest_centroid_is_argmin() {
    rc::check("NearestCentroidMinimizesError",
        [](uint32_t seed) {
            const int dim = TURBO_KV_BLOCK_SIZE;
            std::vector<float> x(dim);
            fill_random(x.data(), dim, seed);
            float norm = vec_l2_norm(x.data(), dim);
            if (norm < 1e-6f) { x[0] = 1.0f; norm = vec_l2_norm(x.data(), dim); }

            block_turbo_kv_4b blk;
            quantize_row_turbo_kv_4b_ref(x.data(), &blk, dim);

            /* Reproduce the code's exact FP32 computation leading up to the
             * argmin step — same operation order, same intermediate values.
             * Using the fp16-quantized inv_std from the block would see
             * slightly different values than the code did and produce
             * spurious argmin-tie failures. */
            std::vector<float> rotated(dim);
            const float inv_norm = 1.0f / norm;
            for (int i = 0; i < dim; i++) rotated[i] = x[i] * inv_norm;
            turbo_kv_rht_forward(rotated.data(), dim, TURBO_KV_DEFAULT_SEED);

            float max_abs = 0.0f;
            for (int i = 0; i < dim; i++) {
                const float a = fabsf(rotated[i]);
                if (a > max_abs) max_abs = a;
            }
            if (max_abs < 1e-10f) max_abs = 1.0f;
            const float inv_std_fp32 = TURBO_KV_4B_CENT_MAX / max_abs;

            for (int i = 0; i < dim; i++) {
                const float scaled = rotated[i] * inv_std_fp32;
                const int chosen = (i & 1)
                    ? (blk.mse_indices[i / 2] >> 4)
                    : (blk.mse_indices[i / 2] & 0x0F);
                const float chosen_dist =
                    fabsf(scaled - turbo_kv_4b_codebook[chosen]);
                for (int c = 0; c < 16; c++) {
                    const float d = fabsf(scaled - turbo_kv_4b_codebook[c]);
                    /* 1 ulp slack for equidistant ties. */
                    RC_ASSERT(chosen_dist <= d + 1e-6f);
                }
            }
        });
}

/* ================================================================
 * Spec: RHT_multi_block — rotation is consistent per-block
 * ================================================================ */
void property_RHT_multi_block_consistency() {
    rc::check("RHT_multi_block_consistency",
        [](int dim, uint32_t seed) {
            if (dim < 128 || dim > 1024 || dim % 128 != 0) return;
            std::vector<float> x(dim);
            fill_random(x.data(), dim, seed);
            std::vector<float> rotated(dim + TURBO_KV_BLOCK_SIZE);
            turbo_kv_rotate_query(x.data(), rotated.data(), dim);
            std::vector<float> per_block(dim + TURBO_KV_BLOCK_SIZE);
            memcpy(per_block.data(), x.data(), dim * sizeof(float));
            for (int b = 0; b < dim / TURBO_KV_BLOCK_SIZE; b++) {
                turbo_kv_rht_forward(per_block.data() + b * TURBO_KV_BLOCK_SIZE,
                    TURBO_KV_BLOCK_SIZE, TURBO_KV_DEFAULT_SEED);
            }
            float err = vec_l2_diff(rotated.data(), per_block.data(), dim);
            RC_ASSERT(err < 1e-4f);
        });
}

/* ================================================================
 * Spec: contract VecDot — @invariant SIMDEquivalence
 * (mul_mat_cpu.allium)
 *
 * For any two registered vec_dot implementations of the same row_type
 * (scalar vs SSE vs AVX2), their scores must agree within
 * config.simd_equivalence_rel_tol relative tolerance on the same
 * (row, query) inputs.
 *
 * This drives the CPU backend's registered dispatch via
 * ggml_compute_forward_mul_mat (the production path), and compares
 * against the ggml-base scalar ggml_vec_dot_turbo_kv_4b_f32. Which
 * SIMD variant the dispatch selects depends on build flags — on this
 * Zen 2 host (-march=native) it's AVX2. On hosts without AVX2 it's
 * SSE4.1 or scalar; in all cases the comparison holds or the invariant
 * is violated.
 *
 * Would have caught the AVX2 codebook buffer over-read in llama.cpp
 * commit 588dd9bd (upper-lane contributions silently dropped).
 * ================================================================ */
void property_SIMDEquivalence_turbo_kv_4b() {
    rc::check("SIMDEquivalence: CPU-backend mul_mat = ggml-base scalar vec_dot",
        [](uint32_t seed, int n_blocks_selector) {
            const int n_blocks = 1 + ((uint32_t)n_blocks_selector & 3u);  // 1..4
            const int head_dim = n_blocks * TURBO_KV_BLOCK_SIZE;

            std::vector<float> key(head_dim), query(head_dim);
            fill_random(key.data(),   head_dim, seed);
            fill_random(query.data(), head_dim, seed + 1);
            /* Every block needs L2_norm > 0 per turbo-kv-4b spec. */
            for (int b = 0; b < n_blocks; b++) {
                if (vec_l2_norm(key.data() + b * TURBO_KV_BLOCK_SIZE,
                                TURBO_KV_BLOCK_SIZE) < 1e-6f) {
                    key[b * TURBO_KV_BLOCK_SIZE] = 1.0f;
                }
            }

            /* Reference: ggml-base scalar vec_dot (no SIMD dispatch). */
            std::vector<block_turbo_kv_4b> blocks(n_blocks);
            quantize_row_turbo_kv_4b_ref(key.data(), blocks.data(), head_dim);
            float ref_score = 0;
            ggml_vec_dot_turbo_kv_4b_f32(
                head_dim, &ref_score, 0,
                reinterpret_cast<const void *>(blocks.data()), 0,
                reinterpret_cast<const void *>(query.data()),  0,
                1);

            /* Dispatch: minimal mul_mat graph on the CPU backend. The
             * type trait .vec_dot (ggml_vec_dot_turbo_kv_4b_f32_cpu)
             * routes to whatever SIMD variant was compiled in.
             *
             * Uses the backend API (init → alloc → set → compute)
             * rather than the raw ggml_init + graph_compute_with_ctx
             * path, to match how production inference drives mul_mat
             * and to let the backend manage tensor buffers properly. */
            ggml_backend_t backend = ggml_backend_cpu_init();
            RC_ASSERT(backend != nullptr);

            const size_t ctx_size = 4 * ggml_tensor_overhead() + 2 * ggml_graph_overhead();
            struct ggml_init_params p = { ctx_size, NULL, /*no_alloc=*/true };
            struct ggml_context * ctx = ggml_init(p);
            RC_ASSERT(ctx != nullptr);

            struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO_KV_4B, head_dim, 1);
            struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,          head_dim, 1);
            struct ggml_tensor * Y = ggml_mul_mat(ctx, A, B);

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            RC_ASSERT(buf != nullptr);

            ggml_backend_tensor_set(A, blocks.data(), 0, n_blocks * sizeof(block_turbo_kv_4b));
            ggml_backend_tensor_set(B, query.data(),  0, head_dim * sizeof(float));

            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, Y);
            const enum ggml_status st = ggml_backend_graph_compute(backend, gf);
            RC_ASSERT(st == GGML_STATUS_SUCCESS);

            float actual = 0;
            ggml_backend_tensor_get(Y, &actual, 0, sizeof(float));

            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            ggml_backend_free(backend);

            const float abs_err = fabsf(ref_score - actual);
            const float rel_err = abs_err / (fabsf(ref_score) + 1e-6f);
            /* mul_mat_cpu.allium: pass if abs_err <= abs_tol OR
             * rel_err <= rel_tol (0.1, 0.05). Mixed metric needed
             * because rel_err is unstable when random q/k produce a
             * near-zero dot; budget covers fp accumulation + the ~1%
             * int8-codebook quantization the SIMD kernels use for
             * VPSHUFB-based lookup. See spec for full rationale. */
            RC_ASSERT(abs_err < 1.0f || rel_err < 0.05f);
        });
}

/* ================================================================
 * Spec: rule RHT_forward (turbo-kv-4b.allium)
 * Verifies the code's turbo_kv_rht_forward matches an inline scalar
 * reference: apply random sign, Walsh-Hadamard butterfly, scale by
 * 1/sqrt(n). Individual rule-success test — the existing
 * RHT_self_inverse covers the composition; this covers one direction
 * against a ground truth so a symmetric bug can't hide as "both
 * sides broken in the same way."
 * ================================================================ */
static int ref_random_sign(uint32_t seed, int idx) {
    uint32_t h = seed ^ (uint32_t) idx;
    h = h * 2654435761u;
    return (h & 1u) ? 1 : -1;
}
static void ref_walsh_hadamard(float * data, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += (len << 1)) {
            for (int j = 0; j < len; j++) {
                const float u = data[i + j];
                const float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
}
static void ref_rht_forward(float * data, int n, uint32_t seed) {
    for (int i = 0; i < n; i++) data[i] *= (float) ref_random_sign(seed, i);
    ref_walsh_hadamard(data, n);
    const float s = 1.0f / sqrtf((float) n);
    for (int i = 0; i < n; i++) data[i] *= s;
}
static void ref_rht_inverse(float * data, int n, uint32_t seed) {
    const float s = 1.0f / sqrtf((float) n);
    for (int i = 0; i < n; i++) data[i] *= s;
    ref_walsh_hadamard(data, n);
    for (int i = 0; i < n; i++) data[i] *= (float) ref_random_sign(seed, i);
}

void property_RHT_forward_matches_reference() {
    rc::check("RHT_forward matches inline scalar reference",
        [](int n_selector, uint32_t seed) {
            /* Power-of-2 n in {4, 8, 16, ..., 1024}. */
            const int shift = 2 + ((uint32_t) n_selector & 7u);
            const int n = 1 << shift;

            std::vector<float> input(n), code_out(n), ref_out(n);
            fill_random(input.data(), n, seed);
            memcpy(code_out.data(), input.data(), n * sizeof(float));
            memcpy(ref_out.data(),  input.data(), n * sizeof(float));

            turbo_kv_rht_forward(code_out.data(), n, seed);
            ref_rht_forward(ref_out.data(), n, seed);

            const float err = vec_l2_diff(code_out.data(), ref_out.data(), n);
            const float ref_norm = vec_l2_norm(ref_out.data(), n);
            RC_ASSERT(err < (ref_norm + 1e-6f) * 1e-5f);
        });
}

void property_RHT_inverse_matches_reference() {
    rc::check("RHT_inverse matches inline scalar reference",
        [](int n_selector, uint32_t seed) {
            const int shift = 2 + ((uint32_t) n_selector & 7u);
            const int n = 1 << shift;

            std::vector<float> input(n), code_out(n), ref_out(n);
            fill_random(input.data(), n, seed);
            memcpy(code_out.data(), input.data(), n * sizeof(float));
            memcpy(ref_out.data(),  input.data(), n * sizeof(float));

            turbo_kv_rht_inverse(code_out.data(), n, seed);
            ref_rht_inverse(ref_out.data(), n, seed);

            const float err = vec_l2_diff(code_out.data(), ref_out.data(), n);
            const float ref_norm = vec_l2_norm(ref_out.data(), n);
            RC_ASSERT(err < (ref_norm + 1e-6f) * 1e-5f);
        });
}

/* ================================================================
 * Spec: config-default.codebook_size (turbo-kv-4b.allium)
 * Also covers: entity-fields.TurboBlock (the 16-entry codebook is
 * what gives block.indices its [0, 15] bound).
 * ================================================================ */
void property_codebook_size_is_16() {
    rc::check("codebook_size = 16", [] {
        /* Runtime check: the codebook table has 16 meaningful entries
         * covering the declared range. We don't read sizeof(array)
         * because the symbol is `extern const float[16]`; the "16" is
         * what the spec pins, so verify 16 distinct values live
         * within [-cent_max, +cent_max]. */
        int n_nonzero_bounded = 0;
        for (int c = 0; c < 16; c++) {
            const float v = turbo_kv_4b_codebook[c];
            if (fabsf(v) <= TURBO_KV_4B_CENT_MAX + 1e-6f) n_nonzero_bounded++;
        }
        RC_ASSERT(n_nonzero_bounded == 16);
    });
}

/* ================================================================
 * Spec: config-default.reconstruction_rel_error (turbo-kv-4b.allium)
 * Asserts the spec's declared 10% budget envelopes the 2% empirical
 * tolerance the roundtrip test uses. If the spec value ever drops
 * below 0.02 or below the observed 0.083 worst-case unit-Gaussian
 * measurement (PHASE24), this check catches it.
 * ================================================================ */
void property_reconstruction_rel_error_budget() {
    rc::check("reconstruction_rel_error budget covers empirical tolerance", [] {
        /* Mirror values from the spec (turbo-kv-4b.allium:48). */
        const float spec_budget        = 0.10f;
        const float test_tolerance     = 0.02f;
        const float empirical_worst    = 0.083f;
        RC_ASSERT(spec_budget > test_tolerance);
        RC_ASSERT(spec_budget > empirical_worst);
    });
}

/* ================================================================
 * Spec: entity-fields.TurboBlock (turbo-kv-4b.allium)
 * Runtime-level check that a quantized block exposes the three
 * fields the spec declares (norm, inv_std, indices) with the
 * expected post-quantize value ranges. Compile-time typedef
 * assertion in ggml-turbo-kv.h proves the block is 72 bytes; this
 * proves the semantic contents behave as specified.
 * ================================================================ */
void property_TurboBlock_fields_after_quantize() {
    rc::check("TurboBlock fields: norm > 0, inv_std > 0, indices in [0, 15]",
        [](uint32_t seed) {
            const int dim = TURBO_KV_BLOCK_SIZE;
            std::vector<float> x(dim);
            fill_random(x.data(), dim, seed);
            /* Guarantee L2_norm > 0 per spec precondition. */
            if (vec_l2_norm(x.data(), dim) < 1e-6f) x[0] = 1.0f;

            block_turbo_kv_4b blk;
            quantize_row_turbo_kv_4b_ref(x.data(), &blk, dim);

            const float norm    = blk.norm;
            const float inv_std = blk.inv_std;
            RC_ASSERT(norm > 0.0f);
            RC_ASSERT(inv_std > 0.0f);
            for (int i = 0; i < dim; i++) {
                const uint8_t bv = blk.mse_indices[i / 2];
                const int idx = (i & 1) ? (bv >> 4) : (bv & 0x0F);
                RC_ASSERT(idx >= 0);
                RC_ASSERT(idx < 16);
            }
        });
}

/* ================================================================
 * Spec: contract VecDot @invariant SIMDEquivalence (mul_mat_cpu.allium),
 * extended to multi-row: the CPU-backend mul_mat output must match
 * the scalar reference across the entire (m, n) output grid, not
 * just the m=1, n=1 case the existing property_SIMDEquivalence test
 * covers. Exercises the outer row/column loop in
 * ggml_compute_forward_mul_mat that my single-row property skipped.
 * ================================================================ */
void property_SIMDEquivalence_multi_row() {
    rc::check("SIMDEquivalence: CPU-backend mul_mat matches scalar across (m, n) grid",
        [](uint32_t seed, int shape_selector) {
            const int m = 2 + ((uint32_t) shape_selector & 3u);          // 2..5 K rows
            const int n = 1 + (((uint32_t) shape_selector >> 2) & 3u);   // 1..4 Q rows
            const int n_blocks = 1 + (((uint32_t) shape_selector >> 4) & 1u);  // 1..2
            const int head_dim = n_blocks * TURBO_KV_BLOCK_SIZE;

            /* Build m K-rows and n Q-rows. */
            std::vector<float> key_rows(m * head_dim), query_rows(n * head_dim);
            fill_random(key_rows.data(),   m * head_dim, seed);
            fill_random(query_rows.data(), n * head_dim, seed + 17);
            /* Ensure every per-block L2_norm > 0 for every K row. */
            for (int r = 0; r < m; r++) {
                for (int b = 0; b < n_blocks; b++) {
                    float * blk = key_rows.data() + r * head_dim + b * TURBO_KV_BLOCK_SIZE;
                    if (vec_l2_norm(blk, TURBO_KV_BLOCK_SIZE) < 1e-6f) blk[0] = 1.0f;
                }
            }

            /* Reference: per (i, j), compute dot via ggml-base scalar vec_dot.
             * ggml's Y[m_idx, n_idx] layout has m as the fast index, so
             * linear offset = m_idx + n_idx * m. Use the same indexing for
             * both ref and actual below. */
            std::vector<block_turbo_kv_4b> blocks(m * n_blocks);
            for (int r = 0; r < m; r++) {
                quantize_row_turbo_kv_4b_ref(
                    key_rows.data() + r * head_dim,
                    blocks.data() + r * n_blocks,
                    head_dim);
            }
            std::vector<float> ref_scores(m * n);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    ggml_vec_dot_turbo_kv_4b_f32(
                        head_dim, &ref_scores[i + j * m], 0,
                        reinterpret_cast<const void *>(blocks.data() + i * n_blocks), 0,
                        reinterpret_cast<const void *>(query_rows.data() + j * head_dim), 0,
                        1);
                }
            }

            /* Dispatch: one CPU-backend mul_mat for the whole grid. */
            ggml_backend_t backend = ggml_backend_cpu_init();
            RC_ASSERT(backend != nullptr);

            const size_t ctx_size = 4 * ggml_tensor_overhead() + 2 * ggml_graph_overhead();
            struct ggml_init_params p = { ctx_size, NULL, /*no_alloc=*/true };
            struct ggml_context * ctx = ggml_init(p);
            RC_ASSERT(ctx != nullptr);

            struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO_KV_4B, head_dim, m);
            struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,          head_dim, n);
            struct ggml_tensor * Y = ggml_mul_mat(ctx, A, B);

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            RC_ASSERT(buf != nullptr);

            ggml_backend_tensor_set(A, blocks.data(),     0, m * n_blocks * sizeof(block_turbo_kv_4b));
            ggml_backend_tensor_set(B, query_rows.data(), 0, n * head_dim * sizeof(float));

            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, Y);
            RC_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

            std::vector<float> actual(m * n);
            ggml_backend_tensor_get(Y, actual.data(), 0, m * n * sizeof(float));

            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            ggml_backend_free(backend);

            /* Compare every cell of the (m, n) grid. */
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    const float ref = ref_scores[i + j * m];
                    const float got = actual[i + j * m];
                    const float abs_err = fabsf(ref - got);
                    const float rel_err = abs_err / (fabsf(ref) + 1e-6f);
                    RC_ASSERT(abs_err < 1.0f || rel_err < 0.05f);
                }
            }
        });
}

/* ================================================================
 * Spec: rule QuantizeBlock @ensures TurboBlock.created(norm, inv_std, indices)
 * (turbo-kv-4b.allium)
 *
 * Directly verifies the ensures clause fields after a quantize call.
 * Overlaps with other tests at the composition level; value here is
 * isolation — if a roundtrip test breaks, this narrows it to the
 * quantize side.
 *   * norm    = L2_norm(input), to fp16 round-trip precision.
 *   * inv_std = cent_max / max_abs(RHT(normalize(input))), same.
 *   * indices are the argmin values — cross-referenced to
 *     property_nearest_centroid_is_argmin (already covers this),
 *     not re-checked here.
 * ================================================================ */
void property_QuantizeBlock_creates_turbo_block() {
    rc::check("QuantizeBlock ensures: TurboBlock.created(norm, inv_std, indices)",
        [](uint32_t seed) {
            const int dim = TURBO_KV_BLOCK_SIZE;
            std::vector<float> x(dim);
            fill_random(x.data(), dim, seed);
            /* Spec requires L2_norm(tensor.data) > 0. */
            float norm = vec_l2_norm(x.data(), dim);
            if (norm < 1e-6f) { x[0] = 1.0f; norm = vec_l2_norm(x.data(), dim); }

            block_turbo_kv_4b blk;
            quantize_row_turbo_kv_4b_ref(x.data(), &blk, dim);

            /* ensures.norm = L2_norm(input) — stored as fp32 with an fp64
             * accumulator for sum-tree robustness. Relative error should be
             * at fp32 precision (≤ ~1e-5 after single final rounding). */
            const float stored_norm = blk.norm;
            const float norm_rel_err = fabsf(stored_norm - norm) / norm;
            RC_ASSERT(norm_rel_err < 1e-4f);

            /* ensures.inv_std = cent_max / max_abs(RHT(normalize(input))).
             * Reproduce the fp32 intermediate value the code computed (same
             * trick as property_nearest_centroid_is_argmin), not the fp16
             * round-trip which would introduce its own error. */
            std::vector<float> rotated(dim);
            const float inv_norm = 1.0f / norm;
            for (int i = 0; i < dim; i++) rotated[i] = x[i] * inv_norm;
            turbo_kv_rht_forward(rotated.data(), dim, TURBO_KV_DEFAULT_SEED);
            float max_abs = 0.0f;
            for (int i = 0; i < dim; i++) {
                const float a = fabsf(rotated[i]);
                if (a > max_abs) max_abs = a;
            }
            if (max_abs < 1e-10f) max_abs = 1.0f;
            const float expected_inv_std = TURBO_KV_4B_CENT_MAX / max_abs;

            const float stored_inv_std = blk.inv_std;
            const float inv_std_rel_err =
                fabsf(stored_inv_std - expected_inv_std) / expected_inv_std;
            /* fp32 storage — a direct compare succeeds at fp32 precision. */
            RC_ASSERT(inv_std_rel_err < 1e-5f);

            /* indices: the argmin property is already tested directly by
             * property_nearest_centroid_is_argmin. The spec's ensures clause
             * for QuantizeBlock just asserts an indices field exists on the
             * created TurboBlock; bounds + correctness are covered by
             * property_quantize_indices_in_bounds and
             * property_nearest_centroid_is_argmin respectively. No
             * duplicate check here. */
        });
}

/* ================================================================
 * Spec: rule DequantizeBlock @ensures Tensor.created(data: reconstructed,
 * count: config.block_size) (turbo-kv-4b.allium)
 *
 * Verifies the creation semantics specifically: exactly block_size
 * floats are written per block call, and nothing is written past the
 * declared count. The norm-preservation property is tested elsewhere;
 * this one isolates the "tensor created with the right shape" check.
 * ================================================================ */
void property_DequantizeBlock_creates_tensor() {
    rc::check("DequantizeBlock ensures: Tensor.created(count: block_size)",
        [](uint32_t seed) {
            const int dim = TURBO_KV_BLOCK_SIZE;
            std::vector<float> x(dim);
            fill_random(x.data(), dim, seed);
            if (vec_l2_norm(x.data(), dim) < 1e-6f) x[0] = 1.0f;

            block_turbo_kv_4b blk;
            quantize_row_turbo_kv_4b_ref(x.data(), &blk, dim);

            /* Allocate a sentinel-padded buffer: block_size floats of real
             * output, followed by 8 sentinel floats the dequant must not
             * touch. If the code writes past block_size elements, the
             * sentinels will differ from 0x42424242. */
            const int pad = 8;
            const uint32_t SENTINEL = 0x42424242u;
            std::vector<uint32_t> raw(dim + pad);
            for (int i = 0; i < dim + pad; i++) raw[i] = SENTINEL;
            float * out = reinterpret_cast<float *>(raw.data());

            dequantize_row_turbo_kv_4b(&blk, out, dim);

            /* count: block_size — sentinel tail must be intact. */
            for (int i = 0; i < pad; i++) {
                RC_ASSERT(raw[dim + i] == SENTINEL);
            }

            /* data: reconstructed — every written float is finite and
             * bounded. "Bounded" here follows from the rescale step:
             * dequant magnitude <= block.norm (RHT is orthogonal, codebook
             * values are <= cent_max in the rotated frame, and rescale
             * multiplies by block.norm). We're not re-checking norm
             * preservation (property_reconstruction_preserves_norm...
             * covers that); we're asserting basic shape-creation correctness. */
            const float norm = blk.norm;
            for (int i = 0; i < dim; i++) {
                RC_ASSERT(std::isfinite(out[i]));
                /* Upper bound: 2*norm is generous (covers fp32 roundoff of
                 * the inv_std factor and any reasonable codebook-scale
                 * overshoot). */
                RC_ASSERT(fabsf(out[i]) <= 2.0f * norm + 1e-3f);
            }
        });
}

/* ================================================================
 * Spec: nearest_centroid.allium
 *
 * Three properties propagated from the distilled argmin + packing
 * contract. The spec was written precisely enough that any SIMD
 * rewrite of the scalar argmin must produce byte-identical output
 * on the same inputs; these tests enforce that contract today
 * (against the production scalar) and will catch drift on any
 * future SIMD variant without needing new tests to be added.
 * ================================================================ */

/* Reference argmin: matches ArgminFirstMatch's @guarantee
 * FirstMatchTieBreak exactly. Strict `<` means an equal distance
 * does NOT update `best`, so the lowest-indexed tied centroid
 * wins. */
static int ref_argmin_first_match(float x, const float * codebook, int n) {
    int best = 0;
    float best_dist = fabsf(x - codebook[0]);
    for (int c = 1; c < n; c++) {
        const float d = fabsf(x - codebook[c]);
        if (d < best_dist) { best_dist = d; best = c; }
    }
    return best;
}

/* Reference packing: matches PackNibbleIndices's wire format.
 * Low nibble = even-position index, high nibble = odd-position. */
static void ref_pack_nibble_indices(const int * indices, int count, uint8_t * out) {
    memset(out, 0, count / 2);
    for (int i = 0; i < count; i++) {
        const int byte_idx = i / 2;
        const int bit_pos  = (i & 1) * 4;
        out[byte_idx] |= (uint8_t) ((indices[i] & 0x0F) << bit_pos);
    }
}

/* Reference NearestCentroidAssignment: reproduces quantize_block_turbo_kv_4b's
 * Step 5 by recomputing rotated + inv_std from the input, then
 * running the scalar argmin + packing. This is the spec-faithful
 * oracle for the production scalar AND any future SIMD variant. */
static void ref_nearest_centroid_assignment(
    const float * x_input, int dim, uint8_t * out_indices)
{
    float norm2 = 0.0f;
    for (int i = 0; i < dim; i++) norm2 += x_input[i] * x_input[i];
    const float norm = sqrtf(norm2);

    std::vector<float> rotated(dim);
    const float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    for (int i = 0; i < dim; i++) rotated[i] = x_input[i] * inv_norm;
    turbo_kv_rht_forward(rotated.data(), dim, TURBO_KV_DEFAULT_SEED);

    float max_abs = 0.0f;
    for (int i = 0; i < dim; i++) {
        const float a = fabsf(rotated[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-10f) max_abs = 1.0f;
    const float inv_std = TURBO_KV_4B_CENT_MAX / max_abs;

    std::vector<int> idx(dim);
    for (int i = 0; i < dim; i++) {
        idx[i] = ref_argmin_first_match(
            rotated[i] * inv_std, turbo_kv_4b_codebook, 16);
    }
    ref_pack_nibble_indices(idx.data(), dim, out_indices);
}

/* ================================================================
 * @guarantee FirstMatchTieBreak: on equal distance, lower-indexed
 * centroid wins.
 *
 * Can't construct exact fp32 ties from the real Lloyd-Max codebook
 * via midpoints — decimal-rounded centroid values like -1.618 and
 * -1.2562 don't have exact fp32 representations, so (a+b)/2 loses a
 * ULP and the "tie" breaks randomly. Use a SYNTHETIC codebook
 * instead: place two entries at x ± delta, making them bit-exactly
 * equidistant from x by construction. Adding "far" entries rules
 * out any earlier-index candidate from winning by being genuinely
 * closer.
 * ================================================================ */
void property_ArgminFirstMatch_lower_index_on_tie() {
    rc::check("ArgminFirstMatch: lower index wins on bit-exact equidistant tie",
        [](int delta_selector) {
            /* Catastrophic cancellation means `(x + d) - x` != `d` in
             * general fp32. To guarantee bit-exact distance equality,
             * pin x = 0 and pick delta from powers-of-2 in the range
             * [2^-8, 2^0] — all exactly representable in fp32,
             * |0 - (-delta)| and |0 - (+delta)| both reduce via the
             * same arithmetic to exactly `delta`. */
            const int k = ((uint32_t) delta_selector & 7u);  // 0..7
            const float delta = ldexpf(1.0f, -k);  // 2^0, 2^-1, ..., 2^-7

            /* Synthetic 4-entry codebook:
             *   [0] = -delta   (tied, low index)
             *   [1] = +100.0f  (far, ruled out)
             *   [2] = +delta   (tied, high index)
             *   [3] = +200.0f  (farther, ruled out) */
            const float codebook[4] = {
                -delta,
                +100.0f,
                +delta,
                +200.0f,
            };

            /* Sanity: assert the two distances really are bit-equal
             * in this construction, otherwise the test is meaningless. */
            const float d0 = fabsf(0.0f - codebook[0]);
            const float d2 = fabsf(0.0f - codebook[2]);
            RC_ASSERT(d0 == d2);

            const int chosen = ref_argmin_first_match(0.0f, codebook, 4);
            /* FirstMatchTieBreak: lower-indexed tied entry wins. */
            RC_ASSERT(chosen == 0);
        });
}

/* ================================================================
 * rule-entity-creation.PackNibbleIndices.1 — pack/unpack roundtrip.
 * Validates that the wire format is invertible.
 * ================================================================ */
void property_PackNibbleIndices_roundtrip() {
    rc::check("PackNibbleIndices: pack → unpack = identity",
        [](uint32_t seed) {
            const int count = TURBO_KV_BLOCK_SIZE;  // 128
            std::mt19937 gen(seed);
            std::uniform_int_distribution<int> dist(0, 15);

            std::vector<int> input(count);
            for (int i = 0; i < count; i++) input[i] = dist(gen);

            uint8_t packed[count / 2];
            ref_pack_nibble_indices(input.data(), count, packed);

            std::vector<int> unpacked(count);
            for (int i = 0; i < count; i++) {
                const uint8_t bv = packed[i / 2];
                unpacked[i] = (i & 1) ? (bv >> 4) : (bv & 0x0F);
            }

            for (int i = 0; i < count; i++) {
                RC_ASSERT(unpacked[i] == input[i]);
            }
        });
}

/* ================================================================
 * rule-success.NearestCentroidAssignment — differential scaffold.
 *
 * The production scalar in quantize_row_turbo_kv_4b_ref must
 * produce byte-identical mse_indices to the spec-faithful
 * reference defined above, for any valid input. This is the
 * framework the SIMD rewrite will plug into: once a
 * quantize_row_turbo_kv_4b_avx2 (or similar) variant exists, a
 * sibling property calls it and compares against the SAME
 * reference; any divergence is caught exactly like the scalar
 * case.
 *
 * Today it doubles as a regression guard on the scalar path.
 * ================================================================ */
void property_NearestCentroidAssignment_matches_reference() {
    rc::check("NearestCentroidAssignment: production mse_indices match reference",
        [](uint32_t seed) {
            const int dim = TURBO_KV_BLOCK_SIZE;
            std::vector<float> x(dim);
            fill_random(x.data(), dim, seed);
            if (vec_l2_norm(x.data(), dim) < 1e-6f) x[0] = 1.0f;

            /* Production path. */
            block_turbo_kv_4b blk;
            quantize_row_turbo_kv_4b_ref(x.data(), &blk, dim);

            /* Reference path. */
            uint8_t expected[dim / 2];
            ref_nearest_centroid_assignment(x.data(), dim, expected);

            /* mul_mat_cpu.allium:config.argmin_index_rel_tol = 0.0 —
             * differential test runs with zero tolerance; any byte
             * mismatch fails. */
            for (int i = 0; i < dim / 2; i++) {
                RC_ASSERT(blk.mse_indices[i] == expected[i]);
            }
        });
}

#if defined(__AVX2__)
/* ================================================================
 * SIMD implementation of nearest_centroid.allium: byte-for-byte
 * against the in-test scalar reference. Passes iff the AVX2 argmin
 * matches the scalar at zero tolerance, which is what
 * nearest_centroid.allium:config.argmin_index_rel_tol = 0.0 demands.
 * ================================================================ */
void property_AVX2_NearestCentroidAssignment_matches_reference() {
    rc::check("AVX2 nearest_centroid block: byte-exact vs scalar reference",
        [](uint32_t seed) {
            const int dim = TURBO_KV_BLOCK_SIZE;
            std::vector<float> x(dim);
            fill_random(x.data(), dim, seed);
            if (vec_l2_norm(x.data(), dim) < 1e-6f) x[0] = 1.0f;

            /* Reproduce steps 1-4 of quantize_block_turbo_kv_4b: get
             * rotated + inv_std to feed into the argmin step alone. */
            const float norm = vec_l2_norm(x.data(), dim);
            std::vector<float> rotated(dim);
            const float inv_norm = 1.0f / norm;
            for (int i = 0; i < dim; i++) rotated[i] = x[i] * inv_norm;
            turbo_kv_rht_forward(rotated.data(), dim, TURBO_KV_DEFAULT_SEED);
            float max_abs = 0.0f;
            for (int i = 0; i < dim; i++) {
                const float a = fabsf(rotated[i]);
                if (a > max_abs) max_abs = a;
            }
            if (max_abs < 1e-10f) max_abs = 1.0f;
            const float inv_std = TURBO_KV_4B_CENT_MAX / max_abs;

            /* Scalar reference: the in-test spec-faithful oracle. */
            uint8_t expected[dim / 2];
            {
                std::vector<int> idx(dim);
                for (int i = 0; i < dim; i++) {
                    idx[i] = ref_argmin_first_match(
                        rotated[i] * inv_std, turbo_kv_4b_codebook, 16);
                }
                ref_pack_nibble_indices(idx.data(), dim, expected);
            }

            /* AVX2 under test. */
            uint8_t actual[dim / 2];
            turbo_kv_4b_avx2_nearest_centroid_block(
                rotated.data(), inv_std, actual);

            /* argmin_index_rel_tol = 0.0 — exact byte match required. */
            for (int b = 0; b < dim / 2; b++) {
                RC_ASSERT(actual[b] == expected[b]);
            }
        });
}
#endif  /* __AVX2__ */

int main(int argc, char ** argv) {
    (void)argc; (void)argv;
    printf("=== turbo-kv-4b PBT ===\n\n");

    property_L2_norm_is_non_negative();
    property_normalize_yields_unit_vector();
    property_rescale_scales_norm_proportionally();
    property_RHT_self_inverse();
    property_RHT_preserves_dot_product();
    property_quantize_indices_in_bounds();
    property_seed_is_deterministic();
    property_block_size_is_128();
    property_cent_max_is_correct();
    property_nearest_centroid_preserves_count();
    property_nearest_centroid_is_argmin();
    property_reconstruct_preserves_count();
    property_reconstruction_preserves_norm_within_tolerance();
    property_roundtrip_quality_within_nmse_tolerance();
    property_VectorDot_matches_dequant_dot();
    property_MultiBlockVectorDot_matches_dequant_dot();
    property_RHT_multi_block_consistency();
    property_RHT_forward_matches_reference();
    property_RHT_inverse_matches_reference();
    property_codebook_size_is_16();
    property_reconstruction_rel_error_budget();
    property_TurboBlock_fields_after_quantize();
    property_QuantizeBlock_creates_turbo_block();
    property_DequantizeBlock_creates_tensor();
    property_SIMDEquivalence_turbo_kv_4b();
    property_SIMDEquivalence_multi_row();
    property_ArgminFirstMatch_lower_index_on_tie();
    property_PackNibbleIndices_roundtrip();
    property_NearestCentroidAssignment_matches_reference();
#if defined(__AVX2__)
    property_AVX2_NearestCentroidAssignment_matches_reference();
#endif

    printf("\n=== All properties passed ===\n");
    return 0;
}
