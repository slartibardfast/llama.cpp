/*
 * turbo_kv_4b_avx2.h — AVX2 kernel for turbo_kv_4b vec_dot
 *
 * Targets Zen 2/3 and any AVX2 CPU.
 *
 * Zen 2 supports _mm256_shuffle_epi8 (VPSHUFB, VEX.256 encoding) natively.
 * The AVX2 kernel processes 32 elements per iteration using 8 XMM
 * accumulators, double the SSE kernel's 16 elements/iter.
 *
 * This file is included from ggml-cpu.c behind #if defined(GGML_AVX2).
 */

#ifndef GGML_TURBO_KV_4B_AVX2_H
#define GGML_TURBO_KV_4B_AVX2_H

#if defined(__AVX2__) && !defined(__ARM_NEON)

#include <immintrin.h>
#include <math.h>

#include "ggml-turbo-kv.h"

/* ============================================================
 * Int8 codebook table — same pattern as SSE kernel
 *
 * The table has 16 entries duplicated to 32 bytes so that a single
 * _mm256_shuffle_epi8 (VPSHUFB) can look up all 32 indices at once.
 * ============================================================ */

/* 32 bytes: 16 int8 centroids duplicated across both 128-bit lanes so a
 * single _mm256_loadu_si256 + _mm256_shuffle_epi8 does correct lookups
 * in BOTH lanes. VPSHUFB treats each 128-bit lane independently, so the
 * upper lane needs its own copy of the table. */
static int8_t turbo_kv_4b_avx2_cb_i8[32] = {0};
static int    turbo_kv_4b_avx2_cb_i8_init = 0;

static inline void turbo_kv_4b_avx2_init_codebook(void) {
    if (turbo_kv_4b_avx2_cb_i8_init) return;
    for (int j = 0; j < 16; j++) {
        const float v = turbo_kv_4b_codebook[j] * (127.0f / TURBO_KV_4B_CENT_MAX);
        int q = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        if (q < -127) q = -127;
        if (q >  127) q =  127;
        turbo_kv_4b_avx2_cb_i8[j]      = (int8_t) q;
        turbo_kv_4b_avx2_cb_i8[j + 16] = (int8_t) q;
    }
    turbo_kv_4b_avx2_cb_i8_init = 1;
}

/* ============================================================
 * AVX2 vec_dot inner kernel — processes 32 elements per iteration
 *
 * Uses native _mm256_shuffle_epi8 (VPSHUFB) for 32-entry table lookup.
 * Zen 2 supports VPSHUFB natively (VEX.256.66.0F38.WIG 00 /r).
 *
 * Returns the accumulated dot product for elements [0..32*full_iters).
 * ============================================================ */
static inline float ggml_vec_dot_turbo_kv_4b_avx2_inner(
    const float * q_rot, const uint8_t * mi, float per_block_scale, int dim)
{
    turbo_kv_4b_avx2_init_codebook();

    /* Load int8 codebook duplicated into a single YMM register (32 bytes = 2×16).
     * _mm256_shuffle_epi8 (VPSHUFB) uses each byte of the indices to select
     * one byte from the table — with 16 entries duplicated, nibbles 0-15
     * map correctly to the right centroid. */
    const __m256i cb_ymm = _mm256_loadu_si256((const __m256i *) turbo_kv_4b_avx2_cb_i8);

    /* Zero-extend mask for int8 → int32 conversion */
    const __m256i mask_0F = _mm256_set1_epi8(0x0F);

    /* Eight XMM accumulators for ILP across the 32 elements per iter */
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    __m128 acc2 = _mm_setzero_ps();
    __m128 acc3 = _mm_setzero_ps();
    __m128 acc4 = _mm_setzero_ps();
    __m128 acc5 = _mm_setzero_ps();
    __m128 acc6 = _mm_setzero_ps();
    __m128 acc7 = _mm_setzero_ps();

    int d = 0;
    for (; d + 31 < dim; d += 32) {
        /* Load 16 bytes of indices (32 nibbles = 32 elements).
         * Zero-extend to 32 bytes for _mm256_shuffle_epi8. */
        __m256i bytes_16;
        {
            const __m128i lo = _mm_loadu_si128((const __m128i *)(mi + d / 2));
            bytes_16 = _mm256_castsi128_si256(lo);
            bytes_16 = _mm256_inserti128_si256(bytes_16, _mm_setzero_si128(), 1);
        }

        /* Split into low/high nibbles (32 values each).
         * _mm256_srli_epi16 shifts 16-bit lanes, so we mask after to
         * drop bleed from the adjacent byte of the low nibble. */
        const __m256i low_nib  = _mm256_and_si256(bytes_16, mask_0F);
        const __m256i high_nib = _mm256_and_si256(
            _mm256_srli_epi16(bytes_16, 4), mask_0F);

        /* Native VPSHUFB: table lookup on each half.
         * Zen 2 supports _mm256_shuffle_epi8 (VEX.256 encoding) natively. */
        const __m256i low_vals  = _mm256_shuffle_epi8(cb_ymm, low_nib);
        const __m256i high_vals = _mm256_shuffle_epi8(cb_ymm, high_nib);

        /* Interleave low/high nibble lookups back into element order.
         * After the two VPSHUFBs, the CORRECT lookups for the 32
         * elements of this iter all live in the LOWER 128-bit lanes of
         * low_vals and high_vals:
         *   low_vals  lower = codebook[even-pos nibbles] — elements
         *     0, 2, ..., 30 (that's 16 lookups, one per byte of
         *     bytes_16's lower lane).
         *   high_vals lower = codebook[odd-pos nibbles] — elements
         *     1, 3, ..., 31.
         * The upper 128-bit lanes are wasted work (bytes_16's upper
         * lane was zero, so they looked up codebook[0] 16 times).
         * `unpacklo` reconstructs elements 0-15 in order; `unpackhi`
         * reconstructs elements 16-31 in order. No need to touch the
         * upper YMM lanes. */
        const __m128i lo_low  = _mm256_castsi256_si128(low_vals);
        const __m128i lo_high = _mm256_castsi256_si128(high_vals);

        const __m128i inter_lo = _mm_unpacklo_epi8(lo_low, lo_high);
        const __m128i inter_hi = _mm_unpackhi_epi8(lo_low, lo_high);

        /* Sign-extend int8 → int32 in four 4-element groups per 128-bit half.
         * _mm_cvtepi8_epi32 reads the low 4 bytes; we shift-right to get
         * the next groups. */
        const __m128i i32_0 = _mm_cvtepi8_epi32(inter_lo);
        const __m128i i32_1 = _mm_cvtepi8_epi32(_mm_srli_si128(inter_lo, 4));
        const __m128i i32_2 = _mm_cvtepi8_epi32(_mm_srli_si128(inter_lo, 8));
        const __m128i i32_3 = _mm_cvtepi8_epi32(_mm_srli_si128(inter_lo, 12));
        const __m128i i32_4 = _mm_cvtepi8_epi32(inter_hi);
        const __m128i i32_5 = _mm_cvtepi8_epi32(_mm_srli_si128(inter_hi, 4));
        const __m128i i32_6 = _mm_cvtepi8_epi32(_mm_srli_si128(inter_hi, 8));
        const __m128i i32_7 = _mm_cvtepi8_epi32(_mm_srli_si128(inter_hi, 12));

        /* int32 → fp32, then apply per-block scale */
        const __m128 scale_v = _mm_set1_ps(per_block_scale);
        const __m128 f0 = _mm_mul_ps(_mm_cvtepi32_ps(i32_0), scale_v);
        const __m128 f1 = _mm_mul_ps(_mm_cvtepi32_ps(i32_1), scale_v);
        const __m128 f2 = _mm_mul_ps(_mm_cvtepi32_ps(i32_2), scale_v);
        const __m128 f3 = _mm_mul_ps(_mm_cvtepi32_ps(i32_3), scale_v);
        const __m128 f4 = _mm_mul_ps(_mm_cvtepi32_ps(i32_4), scale_v);
        const __m128 f5 = _mm_mul_ps(_mm_cvtepi32_ps(i32_5), scale_v);
        const __m128 f6 = _mm_mul_ps(_mm_cvtepi32_ps(i32_6), scale_v);
        const __m128 f7 = _mm_mul_ps(_mm_cvtepi32_ps(i32_7), scale_v);

        /* Mul + add into per-lane accumulators */
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(q_rot + d +  0), f0));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(q_rot + d +  4), f1));
        acc2 = _mm_add_ps(acc2, _mm_mul_ps(_mm_loadu_ps(q_rot + d +  8), f2));
        acc3 = _mm_add_ps(acc3, _mm_mul_ps(_mm_loadu_ps(q_rot + d + 12), f3));
        acc4 = _mm_add_ps(acc4, _mm_mul_ps(_mm_loadu_ps(q_rot + d + 16), f4));
        acc5 = _mm_add_ps(acc5, _mm_mul_ps(_mm_loadu_ps(q_rot + d + 20), f5));
        acc6 = _mm_add_ps(acc6, _mm_mul_ps(_mm_loadu_ps(q_rot + d + 24), f6));
        acc7 = _mm_add_ps(acc7, _mm_mul_ps(_mm_loadu_ps(q_rot + d + 28), f7));
    }

    /* Horizontal reduce: sum 8 accumulators, then reduce the 8 floats. */
    __m128 sum = _mm_add_ps(_mm_add_ps(acc0, acc1),
                            _mm_add_ps(acc2, acc3));
    sum = _mm_add_ps(sum, _mm_add_ps(_mm_add_ps(acc4, acc5),
                                      _mm_add_ps(acc6, acc7)));
    /* hadd twice to collapse to element 0 */
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float mse_dot = _mm_cvtss_f32(sum);

    /* Scalar tail for any remaining elements (dim not a multiple of 32).
     * Uses the same per_block_scale trick — centroid_i8 * per_block_scale
     * equals the dequantized fp32 centroid for this block. */
    if (d < dim) {
        for (; d < dim; d++) {
            const uint8_t bv = mi[d / 2];
            const int idx = (d & 1) ? (bv >> 4) : (bv & 0x0F);
            const float cb_val = (float) turbo_kv_4b_avx2_cb_i8[idx] * per_block_scale;
            mse_dot += q_rot[d] * cb_val;
        }
    }

    /* Caller multiplies by the per-block norm. */
    return mse_dot;
}

/* ============================================================
 * AVX2 block-level wrapper
 *
 * Extracts norm/inv_std, computes per-block scale, calls inner kernel.
 * ============================================================ */
static inline float turbo_kv_4b_avx2_single_block_dot(
    const block_turbo_kv_4b * block,
    const float * q_rot_block,
    int dim_in_block)
{
    const float norm = turbo_kv_fp16_to_fp32(block->norm);
    float inv_std = turbo_kv_fp16_to_fp32(block->inv_std_fp16);
    if (inv_std < 1e-10f) inv_std = sqrtf((float) dim_in_block);
    /* per_block_scale recovers an fp32 centroid from an int8 codebook
     * entry and divides by inv_std:
     *   codebook_fp32[i] = int8_cb[i] * (CENT_MAX / 127)
     *   dequant[i]       = codebook_fp32[i] / inv_std
     * so the inner kernel wants (CENT_MAX / 127) / inv_std. The
     * reciprocal (127/CENT_MAX) scaling here would inflate scores by
     * (127/CENT_MAX)^2 ~ 2160x — caught by the SIMDEquivalence PBT
     * property against the ggml-base scalar path. */
    const float per_block_scale = (TURBO_KV_4B_CENT_MAX / 127.0f) / inv_std;

    return norm * ggml_vec_dot_turbo_kv_4b_avx2_inner(
        q_rot_block, block->mse_indices, per_block_scale, dim_in_block);
}

#define GGML_TURBO_KV_4B_HAVE_AVX2 1

#endif /* __AVX2__ && !__ARM_NEON */

#endif /* GGML_TURBO_KV_4B_AVX2_H */
