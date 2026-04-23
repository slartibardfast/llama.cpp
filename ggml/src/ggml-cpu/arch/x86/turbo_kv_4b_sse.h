/*
 * turbo_kv_4b_sse.h — SSSE3 + SSE4.1 kernel for turbo_kv_4b vec_dot
 *
 * Ports the AVX2 kernel from quantumaikr/quant.cpp v0.8.0 (src/core/tq_turbo_kv.c
 * tq_turbo_kv_4b_attention_ref) to 128-bit SSSE3 + SSE4.1 for Westmere-class x86.
 *
 * The AVX2 version processes 32 elements per iteration with 256-bit accumulators
 * and FMA. This SSSE3 port processes 16 elements per iteration using 128-bit SSE
 * and a mul + add pattern (no FMA on Westmere). Four 128-bit accumulators for ILP,
 * scalar tail for leftover elements.
 *
 * Required ISA: SSSE3 (for _mm_shuffle_epi8, the 16-entry int8 table lookup) and
 * SSE4.1 (for _mm_cvtepi8_epi32 + blendv). Both are Penryn+ (2007) and universal
 * on any Westmere (2010) or newer x86.
 *
 * This file is a header rather than a .c because it's included from
 * ggml-turbo-kv.c behind a feature gate, and the -march=native flag on that
 * compile unit makes the intrinsics available at declaration site.
 */

#ifndef GGML_TURBO_KV_4B_SSE_H
#define GGML_TURBO_KV_4B_SSE_H

#if defined(__SSE4_1__) && defined(__SSSE3__) && !defined(__ARM_NEON)

#include <emmintrin.h>   /* SSE2  */
#include <tmmintrin.h>   /* SSSE3 (_mm_shuffle_epi8) */
#include <smmintrin.h>   /* SSE4.1 (_mm_cvtepi8_epi32) */
#include <string.h>
#include <math.h>

#include "ggml-turbo-kv.h"

/* Pre-scaled int8 codebook: maps each Lloyd-Max centroid c[i] to round(c[i] * 127 / 2.7326).
 *
 * The int8 rescaling loses ~1% precision (step ~0.022 vs typical centroid spacing
 * 0.13-0.66) which is well below our regression threshold (per quant.cpp v0.7.2
 * release notes: "cosine >= 0.99 for 4b"). The gain is a register-resident 16-entry
 * int8 table that fits a single __m128i and can be looked up in one pshufb.
 *
 * The table is initialized at first-call time behind a static guard. Thread-safe
 * because all 16 writes produce the same values regardless of which thread runs
 * first. */
static int8_t turbo_kv_4b_sse_cb_i8[16] = {0};
static int    turbo_kv_4b_sse_cb_i8_init = 0;

static inline void turbo_kv_4b_sse_init_codebook(void) {
    if (turbo_kv_4b_sse_cb_i8_init) return;
    for (int j = 0; j < 16; j++) {
        const float v = turbo_kv_4b_codebook[j] * (127.0f / TURBO_KV_4B_CENT_MAX);
        int q = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        if (q < -127) q = -127;
        if (q >  127) q =  127;
        turbo_kv_4b_sse_cb_i8[j] = (int8_t) q;
    }
    turbo_kv_4b_sse_cb_i8_init = 1;
}

/* ============================================================
 * SSSE3 vec_dot kernel — inner loop only
 *
 * Contract: the caller has already unpacked norm and inv_std from fp16 to
 * fp32 and computed per_block_scale = (2.7326/127) / inv_std. This keeps
 * this header free of any fp16 conversion, which would otherwise require
 * pulling in simd-mappings.h from ggml-cpu. The caller also handles the
 * scalar tail for dim not a multiple of 16.
 *
 * Returns the accumulated dot product for elements [0..16*full_iters).
 * ============================================================ */
static inline float ggml_vec_dot_turbo_kv_4b_sse_inner(
    const float * q_rot, const uint8_t * mi, float per_block_scale, int dim)
{
    turbo_kv_4b_sse_init_codebook();

    const __m128 scale_v = _mm_set1_ps(per_block_scale);

    /* Load int8 codebook into a single XMM register */
    const __m128i cb_xmm = _mm_loadu_si128((const __m128i *) turbo_kv_4b_sse_cb_i8);
    const __m128i mask0F = _mm_set1_epi8(0x0F);

    /* Four 128-bit accumulators for ILP across the 16 elements per iter */
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    __m128 acc2 = _mm_setzero_ps();
    __m128 acc3 = _mm_setzero_ps();

    int d = 0;
    for (; d + 15 < dim; d += 16) {
        /* Load 8 bytes of indices (16 nibbles = 16 elements).
         * The loaded data is in the low 8 bytes; high 8 bytes are zero. */
        const __m128i bytes_8 = _mm_loadl_epi64((const __m128i *)(mi + d / 2));

        /* Split into low/high nibbles.
         *
         * For the low nibble we just mask: low[i] = byte[i] & 0x0F.
         * For the high nibble we shift-right then mask: high[i] = (byte[i] >> 4).
         * _mm_srli_epi16 shifts 16-bit lanes so we need the mask afterwards to
         * drop bleed from the adjacent byte. */
        const __m128i low_nib  = _mm_and_si128(bytes_8, mask0F);
        const __m128i high_nib = _mm_and_si128(_mm_srli_epi16(bytes_8, 4), mask0F);

        /* 16-entry int8 table lookup via SSSE3 pshufb. Each lane's low nibble
         * indexes the cb_xmm register and returns the corresponding int8 value. */
        const __m128i low_vals  = _mm_shuffle_epi8(cb_xmm, low_nib);
        const __m128i high_vals = _mm_shuffle_epi8(cb_xmm, high_nib);

        /* Interleave so that element [2*i] = low_vals[i], [2*i+1] = high_vals[i].
         * unpacklo_epi8 does exactly this on the first 8 int8 pairs.
         * The result holds the 16 int8 reconstructed centroid values for
         * elements d..d+15. */
        const __m128i inter = _mm_unpacklo_epi8(low_vals, high_vals);

        /* Convert int8 -> int32 in four 4-element groups, then to fp32.
         * _mm_cvtepi8_epi32 reads the low 4 bytes of its input. We shift-right
         * by 4, 8, 12 bytes to get the next groups. */
        const __m128i i32_0 = _mm_cvtepi8_epi32(inter);
        const __m128i i32_1 = _mm_cvtepi8_epi32(_mm_srli_si128(inter, 4));
        const __m128i i32_2 = _mm_cvtepi8_epi32(_mm_srli_si128(inter, 8));
        const __m128i i32_3 = _mm_cvtepi8_epi32(_mm_srli_si128(inter, 12));

        /* int32 -> fp32, then apply per-block scale. */
        const __m128 f0 = _mm_mul_ps(_mm_cvtepi32_ps(i32_0), scale_v);
        const __m128 f1 = _mm_mul_ps(_mm_cvtepi32_ps(i32_1), scale_v);
        const __m128 f2 = _mm_mul_ps(_mm_cvtepi32_ps(i32_2), scale_v);
        const __m128 f3 = _mm_mul_ps(_mm_cvtepi32_ps(i32_3), scale_v);

        /* Mul + add (no FMA on Westmere) into the per-lane accumulators. */
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(q_rot + d +  0), f0));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(q_rot + d +  4), f1));
        acc2 = _mm_add_ps(acc2, _mm_mul_ps(_mm_loadu_ps(q_rot + d +  8), f2));
        acc3 = _mm_add_ps(acc3, _mm_mul_ps(_mm_loadu_ps(q_rot + d + 12), f3));
    }

    /* Horizontal reduce: sum 4 accumulators, then reduce the 4 floats. */
    __m128 sum = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
    /* hadd twice to collapse to element 0 */
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float mse_dot = _mm_cvtss_f32(sum);

    /* Scalar tail for any remaining elements (dim not a multiple of 16).
     * Uses the same per_block_scale trick — centroid_i8 * per_block_scale
     * equals the dequantized fp32 centroid for this block. */
    if (d < dim) {
        for (; d < dim; d++) {
            const uint8_t bv = mi[d / 2];
            const int idx = (d & 1) ? (bv >> 4) : (bv & 0x0F);
            const float cb_val = (float) turbo_kv_4b_sse_cb_i8[idx] * per_block_scale;
            mse_dot += q_rot[d] * cb_val;
        }
    }

    /* Caller multiplies by the per-block norm. */
    return mse_dot;
}

/* ============================================================
 * SSE block-level wrapper
 *
 * Extracts norm/inv_std, computes per-block scale, calls inner kernel.
 * ============================================================ */
static inline float turbo_kv_4b_sse_single_block_dot(
    const block_turbo_kv_4b * block,
    const float * q_rot_block,
    int dim_in_block)
{
    const float norm = turbo_kv_fp16_to_fp32(block->norm);
    float inv_std = turbo_kv_fp16_to_fp32(block->inv_std_fp16);
    if (inv_std < 1e-10f) inv_std = sqrtf((float) dim_in_block);
    const float scale = 1.0f / inv_std;
    const float per_block_scale = (127.0f / TURBO_KV_4B_CENT_MAX) * scale;

    return norm * ggml_vec_dot_turbo_kv_4b_sse_inner(
        q_rot_block, block->mse_indices, per_block_scale, dim_in_block);
}

#define GGML_TURBO_KV_4B_HAVE_SSE 1

#endif /* __SSE4_1__ && __SSSE3__ */

#endif /* GGML_TURBO_KV_4B_SSE_H */
