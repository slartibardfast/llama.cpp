/*
 * downlevel.h — hand-rolled SSE4.1 implementations of AVX / AVX2 / F16C
 * intrinsics that ggml hot loops need on Westmere-class x86.
 *
 * Layer 1 in the layered shimming architecture (see
 * serene-bubbling-orbit.md). Each function here is:
 *   - register-resident (no stack spills)
 *   - static inline (no linker cost, inlined at the call site)
 *   - cited to a public-domain / Apache 2.0 / original source
 *   - tested for correctness against a scalar reference
 *
 * Callers explicitly write `ggml_x86_*` at the call site; we do NOT
 * alias the `_mm*` intrinsic namespace. On AVX-capable hosts these
 * helpers are still available and produce identical results, but the
 * caller should prefer the real intrinsic directly (we provide a
 * GGML_X86_HAS_NATIVE_AVX define to signal that).
 */

#ifndef GGML_X86_DOWNLEVEL_H
#define GGML_X86_DOWNLEVEL_H

#include <stdint.h>

#if !defined(__SSE4_1__)
#error "downlevel.h requires SSE4.1 (Penryn / Westmere or newer)"
#endif

#include <emmintrin.h>   /* SSE2 */
#include <tmmintrin.h>   /* SSSE3 */
#include <smmintrin.h>   /* SSE4.1 */
#include <nmmintrin.h>   /* SSE4.2 (for _mm_popcnt_u64 where available) */

/* LTO + per-source -march=native interaction:
 *
 * ggml-turbo-quant.c and ggml-cpu.c are both compiled with -march=native
 * via set_source_files_properties/target flags, which enables SSE4.1 and
 * POPCNT on the Westmere target. With LTO, however, the link-time
 * optimiser may inline functions from these TUs into callers in
 * ggml-base (which is compiled without -march=native). That re-
 * compilation strips the per-TU target attribute and can cause
 * intrinsic emission to fail, or worse, silently fall back to a
 * scalar-ish sequence.
 *
 * The fix: annotate our helpers with __attribute__((target(...)))
 * so the function-level target survives LTO inlining. GCC and Clang
 * both honour this at LTO time. */
#if defined(__GNUC__) && !defined(__clang__)
#define GGML_X86_TARGET_DOWNLEVEL __attribute__((target("sse2,ssse3,sse4.1,sse4.2,popcnt")))
#else
/* Clang does not need per-function target when -march=native is set
 * on the TU; its LTO preserves target attributes by default. */
#define GGML_X86_TARGET_DOWNLEVEL
#endif

#if defined(__AVX__) || defined(__F16C__)
#define GGML_X86_HAS_NATIVE_AVX 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Half-precision <-> single-precision conversion
 *
 * Vendored from Fabian Giesen's "Half to float done quic" blog post:
 *   https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
 *   https://gist.github.com/rygorous/2156668
 * License: public domain per the gist header.
 *
 * Algorithm:
 *   - Mask sign / exponent / mantissa fields
 *   - Normal path: bias-shift exponent from [-14, 15] to [-126, 127]
 *     (equivalently, add ((127 - 15) << 23) to the shifted exp16)
 *   - Denormal path: a "magic" fp32 multiply renormalises the value
 *   - Infinity / NaN: exp == 0x7C00 -> force exp32 to 0xFF000000
 *
 * ~19 SSE2 instructions per 4-element conversion, fully register
 * resident. On a 2.66 GHz Westmere core this costs ~8 cycles per
 * 4-element batch on a normal-value workload.
 * ============================================================ */

/* 4 fp16 values in the low 8 bytes of the input → 4 fp32 values.
 * The upper 8 bytes of the input are ignored. */
GGML_X86_TARGET_DOWNLEVEL
static inline __m128 ggml_x86_cvtph_ps(__m128i h) {
#if defined(__F16C__)
    /* On native F16C hosts we hand off to the hardware instruction
     * so that this header is a drop-in on any x86 target. */
    return _mm_cvtph_ps(h);
#else
    /* Constants, following Giesen's half-to-float SSE2 gist. */
    const __m128i mask_nosign   = _mm_set1_epi32(0x7FFF);
    const __m128i mask_justsign = _mm_set1_epi32(0x8000);
    const __m128i was_infnan    = _mm_set1_epi32(0x7BFF);   /* threshold: > this means inf/NaN */
    const __m128  magic         = _mm_castsi128_ps(_mm_set1_epi32((254 - 15) << 23));
    const __m128i inf_exp_fp32  = _mm_set1_epi32(0x7F800000); /* fp32 +inf exponent pattern */

    /* Zero-extend the 4 packed fp16 values to 32-bit lanes */
    __m128i expmant_full = _mm_cvtepu16_epi32(h);

    /* Split sign and magnitude (15-bit) */
    __m128i sign    = _mm_and_si128(expmant_full, mask_justsign);
    __m128i expmant = _mm_and_si128(expmant_full, mask_nosign);

    /* Shift sign to bit 31 and magnitude up 13 bits */
    __m128i sign32  = _mm_slli_epi32(sign, 16);
    __m128i shifted = _mm_slli_epi32(expmant, 13);

    /* Multiply by magic (2^(254-15-127) = 2^112) to rebias the exponent
     * from fp16 (bias 15) to fp32 (bias 127). This conveniently
     * renormalises fp16 denormals as a side effect. */
    __m128  scaled  = _mm_mul_ps(_mm_castsi128_ps(shifted), magic);

    /* Special case: fp16 inf/NaN when expmant > 0x7BFF (i.e. exp == 0x1F).
     * We need to OR in 0x7F800000 to bump the fp32 exponent to 0xFF. The
     * mantissa carried through from `shifted` preserves NaN payloads. */
    __m128i b_was_infnan = _mm_cmpgt_epi32(expmant, was_infnan);
    __m128i inf_patch    = _mm_and_si128(b_was_infnan, inf_exp_fp32);

    /* Assemble final: scaled-or-inf_patch OR sign32 */
    __m128i result = _mm_or_si128(_mm_castps_si128(scaled), inf_patch);
    result         = _mm_or_si128(result, sign32);

    return _mm_castsi128_ps(result);
#endif
}

/* 8 fp16 → 8 fp32. Writes 8 floats to `out`. */
GGML_X86_TARGET_DOWNLEVEL
static inline void ggml_x86_cvtph_ps_8(const void * src_fp16_bytes, float * out) {
    __m128i h = _mm_loadu_si128((const __m128i *) src_fp16_bytes);
#if defined(__F16C__)
    /* Single AVX instruction on F16C hosts, 8 fp16 → 8 fp32 */
    __m256 y = _mm256_cvtph_ps(h);
    _mm256_storeu_ps(out, y);
#else
    /* Westmere: two calls to the 4-element helper, split on the low/high
     * 64-bit halves of the 128-bit load. */
    __m128 lo = ggml_x86_cvtph_ps(h);
    __m128 hi = ggml_x86_cvtph_ps(_mm_srli_si128(h, 8));
    _mm_storeu_ps(out,     lo);
    _mm_storeu_ps(out + 4, hi);
#endif
}

/* ============================================================
 * q4_0 dequantize (32 elements per block, SSE4.1 native)
 *
 * Original, informed by ggml's reference scalar loop in
 * ggml/src/ggml-quants.c dequantize_row_q4_0. Uses PMOVZXBD (SSE4.1)
 * to expand nibble groups to int32 lanes with no memory traffic.
 * Register-resident; ~30 SSE instructions per 32-element block.
 * ============================================================ */

/* Unpack 32 nibbles (16 bytes) + fp32 scale into 32 floats.
 * qs must point to 16 bytes of packed nibbles, low nibble first.
 * Output layout matches the ggml-base reference:
 *   out[0..15] come from low nibbles (qs[0..15] & 0x0F) - 8
 *   out[16..31] come from high nibbles (qs[0..15] >> 4) - 8
 * both scaled by `scale`. */
GGML_X86_TARGET_DOWNLEVEL
static inline void ggml_x86_q4_0_unpack_32(const uint8_t * qs, float scale, float * out) {
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i bias8   = _mm_set1_epi32(8);
    const __m128  vd      = _mm_set1_ps(scale);

    const __m128i packed = _mm_loadu_si128((const __m128i *) qs);

    /* Low nibbles */
    const __m128i lo = _mm_and_si128(packed, mask_0f);
    __m128i l0 = _mm_sub_epi32(_mm_cvtepu8_epi32(lo),                            bias8);
    __m128i l1 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(lo, 4)),         bias8);
    __m128i l2 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(lo, 8)),         bias8);
    __m128i l3 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(lo, 12)),        bias8);
    _mm_storeu_ps(out + 0,  _mm_mul_ps(_mm_cvtepi32_ps(l0), vd));
    _mm_storeu_ps(out + 4,  _mm_mul_ps(_mm_cvtepi32_ps(l1), vd));
    _mm_storeu_ps(out + 8,  _mm_mul_ps(_mm_cvtepi32_ps(l2), vd));
    _mm_storeu_ps(out + 12, _mm_mul_ps(_mm_cvtepi32_ps(l3), vd));

    /* High nibbles. srli_epi16 is fine because we mask with 0x0F after,
     * which discards the bleed from the low nibble of the adjacent byte. */
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_0f);
    __m128i h0 = _mm_sub_epi32(_mm_cvtepu8_epi32(hi),                            bias8);
    __m128i h1 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(hi, 4)),         bias8);
    __m128i h2 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(hi, 8)),         bias8);
    __m128i h3 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(hi, 12)),        bias8);
    _mm_storeu_ps(out + 16, _mm_mul_ps(_mm_cvtepi32_ps(h0), vd));
    _mm_storeu_ps(out + 20, _mm_mul_ps(_mm_cvtepi32_ps(h1), vd));
    _mm_storeu_ps(out + 24, _mm_mul_ps(_mm_cvtepi32_ps(h2), vd));
    _mm_storeu_ps(out + 28, _mm_mul_ps(_mm_cvtepi32_ps(h3), vd));
}

#ifdef __cplusplus
}
#endif

#endif /* GGML_X86_DOWNLEVEL_H */
