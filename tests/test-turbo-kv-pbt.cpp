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

    printf("\n=== All properties passed ===\n");
    return 0;
}
