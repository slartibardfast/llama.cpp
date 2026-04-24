/*
 * ggml-turbo-kv.c — scalar C reference for turbo_kv_4b
 *
 * Ported from quantumaikr/quant.cpp v0.8.0 (Apache-2.0).
 *
 * This file contains:
 *   - The 16-entry Lloyd-Max-Gaussian codebook table
 *   - Random Hadamard Transform (forward + inverse) using an O(n log n)
 *     butterfly and a Knuth-hash-based random-sign diagonal.
 *   - quantize_row_turbo_kv_4b_ref and dequantize_row_turbo_kv_4b, operating
 *     block-at-a-time (128 elements in, 72 bytes out or vice versa).
 *   - turbo_kv_rotate_query: the Q-side preprocessing that happens once per
 *     attention row.
 *   - ggml_vec_dot_turbo_kv_4b_f32: the scalar fallback K dot used by the
 *     type-traits table. SIMD overrides live in arch-specific headers.
 *
 * The SSSE3 (_mm_shuffle_epi8) kernel for Westmere lives in
 * ggml/src/ggml-cpu/arch/x86/turbo_kv_4b_sse.h and overrides this scalar
 * path via the type_traits_cpu registration.
 */

#include "ggml-turbo-kv.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <float.h>

/* ============================================================
 * Lloyd-Max-Gaussian 16-entry codebook for N(0,1)
 * ============================================================ */

const float turbo_kv_4b_codebook[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9423f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9423f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f
};

/* Published Lloyd-Max Gaussian codebooks from tq_codebook.c (Max 1960).
 * Weight quantization uses the SAME codebooks as KV cache — the post-RHT
 * distribution is confirmed near-Gaussian (kurtosis 2.96, KS 0.003).
 * Model-specific codebooks can be computed via llama-turbo-codebook tool
 * and loaded at quantize time via --codebook flag. */

const float turbo_codebook_2bit[4] = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f
};

const float turbo_codebook_3bit[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3440f,  2.1520f
};

/* turbo_kv_4b_codebook[16] is the 4-bit codebook — already defined above */

const float turbo_codebook_5bit[32] = {
    -1.9956f, -1.7900f, -1.6107f, -1.4493f, -1.3010f, -1.1631f, -1.0334f, -0.9104f,
    -0.7928f, -0.6795f, -0.5697f, -0.4626f, -0.3576f, -0.2543f, -0.1520f, -0.0506f,
     0.0506f,  0.1520f,  0.2543f,  0.3576f,  0.4626f,  0.5697f,  0.6795f,  0.7928f,
     0.9104f,  1.0334f,  1.1631f,  1.3010f,  1.4493f,  1.6107f,  1.7900f,  1.9956f
};

/* Max centroid magnitudes for per-block scaling: inv_std = CENT_MAX / max_abs */
#define TURBO_2B_CENT_MAX 1.5104f
#define TURBO_3B_CENT_MAX 2.1520f
#define TURBO_5B_CENT_MAX 1.9956f
/* 4-bit uses TURBO_KV_4B_CENT_MAX (2.7326f) from the header */

/* Per-centroid decision boundaries are midpoints between consecutive
 * centroids. For nearest-centroid lookup we compute |x - c| and take min.
 * With only 4-32 entries, linear scan beats binary search. */

/* ============================================================
 * Random sign generation from seed (Knuth multiplicative hash)
 * ============================================================ */

static inline int turbo_kv_random_sign(uint32_t seed, int idx) {
    uint32_t h = seed ^ (uint32_t) idx;
    h = h * 2654435761u;
    return (h & 1u) ? 1 : -1;
}

/* ============================================================
 * In-place Walsh-Hadamard butterfly (unnormalized).
 *
 * For len = 1, 2, 4, ..., n/2:
 *   for each block of 2*len elements:
 *     butterfly the pair (u[j], u[j+len]) -> (u+v, u-v)
 *
 * n must be power of 2. Cost: n * log2(n) adds/subs.
 * ============================================================ */
static void walsh_hadamard(float * data, int n) {
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

#ifdef __SSE4_1__
/* SSE4.1 Walsh-Hadamard butterfly. n must be power of 2, >= 4.
 *
 * Stage 1 (stride 1): [u0,v0,u1,v1] -> [u0+v0, u0-v0, u1+v1, u1-v1]
 *   hadd/hsub give horizontal add/sub, unpacklo interleaves them.
 *
 * Stage 2 (stride 2): [a,b,c,d] -> [a+c, b+d, a-c, b-d]
 *   movelh/movehl split low/high halves, add/sub, shuffle to recombine.
 *
 * Stage 3+ (stride >= 4): contiguous 4-wide load, add, sub, store. */
static void walsh_hadamard_sse(float * data, int n) {
    for (int i = 0; i < n; i += 4) {
        __m128 a = _mm_loadu_ps(&data[i]);
        _mm_storeu_ps(&data[i], _mm_unpacklo_ps(_mm_hadd_ps(a, a),
                                                 _mm_hsub_ps(a, a)));
    }
    for (int i = 0; i < n; i += 4) {
        __m128 a  = _mm_loadu_ps(&data[i]);
        __m128 lo = _mm_movelh_ps(a, a);
        __m128 hi = _mm_movehl_ps(a, a);
        _mm_storeu_ps(&data[i], _mm_shuffle_ps(_mm_add_ps(lo, hi),
                                                _mm_sub_ps(lo, hi),
                                                _MM_SHUFFLE(1, 0, 1, 0)));
    }
    for (int len = 4; len < n; len <<= 1) {
        for (int i = 0; i < n; i += (len << 1)) {
            for (int j = 0; j < len; j += 4) {
                __m128 u = _mm_loadu_ps(&data[i + j]);
                __m128 v = _mm_loadu_ps(&data[i + j + len]);
                _mm_storeu_ps(&data[i + j],       _mm_add_ps(u, v));
                _mm_storeu_ps(&data[i + j + len], _mm_sub_ps(u, v));
            }
        }
    }
}
#endif

/* Round n down to the nearest power of 2 (n2 <= n, n2 is a power of 2). */
static inline int turbo_kv_pow2_floor(int n) {
    int n2 = 1;
    while ((n2 << 1) <= n) n2 <<= 1;
    return n2;
}

void turbo_kv_rht_forward(float * data, int n, uint32_t seed) {
    if (!data || n <= 0) return;
    const int n2 = turbo_kv_pow2_floor(n);

#ifdef __SSE4_1__
    /* Vectorized sign flip + scale: XOR sign bits, mul by 1/sqrt(n2) */
    if (n2 >= 4) {
        for (int i = 0; i < n2; i += 4) {
            __m128 d = _mm_loadu_ps(&data[i]);
            uint32_t smask[4] __attribute__((aligned(16)));
            for (int k = 0; k < 4; k++) {
                smask[k] = turbo_kv_random_sign(seed, i+k) < 0 ? 0x80000000u : 0;
            }
            _mm_storeu_ps(&data[i], _mm_xor_ps(d, _mm_load_ps((const float*)smask)));
        }
        walsh_hadamard_sse(data, n2);
        const __m128 vscale = _mm_set1_ps(1.0f / sqrtf((float) n2));
        for (int i = 0; i < n2; i += 4) {
            _mm_storeu_ps(&data[i], _mm_mul_ps(_mm_loadu_ps(&data[i]), vscale));
        }
        return;
    }
#endif
    for (int i = 0; i < n2; i++) {
        data[i] *= (float) turbo_kv_random_sign(seed, i);
    }
    walsh_hadamard(data, n2);
    const float scale = 1.0f / sqrtf((float) n2);
    for (int i = 0; i < n2; i++) {
        data[i] *= scale;
    }
}

void turbo_kv_rht_inverse(float * data, int n, uint32_t seed) {
    if (!data || n <= 0) return;
    const int n2 = turbo_kv_pow2_floor(n);

#ifdef __SSE4_1__
    if (n2 >= 4) {
        const __m128 vscale = _mm_set1_ps(1.0f / sqrtf((float) n2));
        for (int i = 0; i < n2; i += 4) {
            _mm_storeu_ps(&data[i], _mm_mul_ps(_mm_loadu_ps(&data[i]), vscale));
        }
        walsh_hadamard_sse(data, n2);
        for (int i = 0; i < n2; i += 4) {
            __m128 d = _mm_loadu_ps(&data[i]);
            uint32_t smask[4] __attribute__((aligned(16)));
            for (int k = 0; k < 4; k++) {
                smask[k] = turbo_kv_random_sign(seed, i+k) < 0 ? 0x80000000u : 0;
            }
            _mm_storeu_ps(&data[i], _mm_xor_ps(d, _mm_load_ps((const float*)smask)));
        }
        return;
    }
#endif
    const float scale = 1.0f / sqrtf((float) n2);
    for (int i = 0; i < n2; i++) {
        data[i] *= scale;
    }
    walsh_hadamard(data, n2);
    for (int i = 0; i < n2; i++) {
        data[i] *= (float) turbo_kv_random_sign(seed, i);
    }
}

/* ============================================================
 * fp16 / fp32 conversion helpers.
 *
 * We use ggml's existing FP16_TO_FP32 / FP32_TO_FP16 macros via
 * ggml-impl.h so we don't introduce a second conversion path.
 * ============================================================ */
float turbo_kv_fp16_to_fp32(uint16_t h) {
    ggml_fp16_t h16;
    memcpy(&h16, &h, sizeof(h16));
    return GGML_FP16_TO_FP32(h16);
}

uint16_t turbo_kv_fp32_to_fp16(float f) {
    const ggml_fp16_t r = GGML_FP32_TO_FP16(f);
    uint16_t out;
    memcpy(&out, &r, sizeof(out));
    return out;
}

/* ============================================================
 * Quantize one block: steps 1-4 (L2 norm, normalize, RHT, inv_std)
 *
 * Exposed so ggml-cpu's AVX2 quantize can share the same prep pass
 * as the scalar reference. The scalar `quantize_block_turbo_kv_4b`
 * below calls this + the scalar Step 5; the AVX2 variant in ggml-cpu
 * calls this + `turbo_kv_4b_avx2_nearest_centroid_block`. Keeping
 * the shared prep in one place means future pipeline changes (new
 * seed, different norm clamp, etc.) don't need to be mirrored.
 *
 * Writes:
 *   block->norm, block->inv_std_fp16, block->residual_norm, block->_pad
 *   rotated_out[0..TURBO_KV_BLOCK_SIZE-1] (filled; slack zero-padded)
 * Returns inv_std as fp32 (so the caller's Step 5 can use the full-
 * precision value, not the fp16 round-trip that would lose a ULP
 * and potentially flip argmin at tie boundaries).
 * ============================================================ */
float turbo_kv_4b_prepare_block(
    const float * src, int dim, block_turbo_kv_4b * block, float * rotated_out)
{
    if (dim > TURBO_KV_BLOCK_SIZE) dim = TURBO_KV_BLOCK_SIZE;

    /* Step 1: L2 norm */
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm_sq += src[i] * src[i];
    }
    const float norm = sqrtf(norm_sq);
    block->norm = turbo_kv_fp32_to_fp16(norm);
    block->residual_norm = 0;
    block->_pad = 0;

    /* Step 2: normalize to unit vector (slack zero-padded) */
    const float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    for (int i = 0; i < dim; i++) {
        rotated_out[i] = src[i] * inv_norm;
    }
    for (int i = dim; i < TURBO_KV_BLOCK_SIZE; i++) {
        rotated_out[i] = 0.0f;
    }

    /* Step 3: RHT (in-place) */
    turbo_kv_rht_forward(rotated_out, dim, TURBO_KV_DEFAULT_SEED);

    /* Step 4: compute max_abs and per-block inv_std. Max is associative
     * for non-NaN fp32, so SIMD reduction is bit-exact with the scalar
     * sequential max. Our inputs are bounded in [-cent_max, cent_max]
     * after RHT so non-NaN is guaranteed. */
    float max_abs = 0.0f;
#if defined(__AVX2__)
    {
        const __m256 sign_clear =
            _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        __m256 max_vec = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < dim; i += 8) {
            const __m256 v = _mm256_loadu_ps(rotated_out + i);
            const __m256 abs_v = _mm256_and_ps(v, sign_clear);
            max_vec = _mm256_max_ps(max_vec, abs_v);
        }
        /* Horizontal max of 8 lanes. */
        float tmp[8];
        _mm256_storeu_ps(tmp, max_vec);
        for (int k = 0; k < 8; k++) {
            if (tmp[k] > max_abs) max_abs = tmp[k];
        }
        /* Scalar tail for dim not a multiple of 8. */
        for (; i < dim; i++) {
            const float a = fabsf(rotated_out[i]);
            if (a > max_abs) max_abs = a;
        }
    }
#else
    for (int i = 0; i < dim; i++) {
        const float a = fabsf(rotated_out[i]);
        if (a > max_abs) max_abs = a;
    }
#endif
    if (max_abs < 1e-10f) max_abs = 1.0f;
    const float inv_std = TURBO_KV_4B_CENT_MAX / max_abs;
    block->inv_std_fp16 = turbo_kv_fp32_to_fp16(inv_std);

    return inv_std;
}

/* ============================================================
 * Quantize one block (128 elements -> 72 bytes), scalar Step 5.
 * Shares Steps 1-4 with the AVX2 variant via turbo_kv_4b_prepare_block.
 * ============================================================ */
static void quantize_block_turbo_kv_4b(const float * src, block_turbo_kv_4b * block, int dim) {
    if (dim > TURBO_KV_BLOCK_SIZE) dim = TURBO_KV_BLOCK_SIZE;

    float rotated[TURBO_KV_BLOCK_SIZE];
    const float inv_std = turbo_kv_4b_prepare_block(src, dim, block, rotated);

    /* Step 5: per-element nearest-centroid lookup after scaling by inv_std */
    memset(block->mse_indices, 0, TURBO_KV_BLOCK_SIZE / 2);
    for (int i = 0; i < dim; i++) {
        const float x = rotated[i] * inv_std;
        int best = 0;
        float best_dist = fabsf(x - turbo_kv_4b_codebook[0]);
        for (int c = 1; c < 16; c++) {
            const float d = fabsf(x - turbo_kv_4b_codebook[c]);
            if (d < best_dist) {
                best_dist = d;
                best = c;
            }
        }
        const int byte_idx = i / 2;
        const int bit_pos  = (i & 1) * 4;
        block->mse_indices[byte_idx] |= (uint8_t) ((best & 0x0F) << bit_pos);
    }
}

void quantize_row_turbo_kv_4b_ref(const float * x, block_turbo_kv_4b * y, int64_t k) {
    GGML_ASSERT(k % TURBO_KV_BLOCK_SIZE == 0);
    const int64_t nb = k / TURBO_KV_BLOCK_SIZE;
    for (int64_t b = 0; b < nb; b++) {
        quantize_block_turbo_kv_4b(x + b * TURBO_KV_BLOCK_SIZE, y + b, TURBO_KV_BLOCK_SIZE);
    }
}

/* ============================================================
 * Dequantize one block (72 bytes -> 128 elements)
 * ============================================================ */
static void dequantize_block_turbo_kv_4b(const block_turbo_kv_4b * block, float * dst, int dim) {
    if (dim > TURBO_KV_BLOCK_SIZE) dim = TURBO_KV_BLOCK_SIZE;

    const float norm = turbo_kv_fp16_to_fp32(block->norm);
    float inv_std = turbo_kv_fp16_to_fp32(block->inv_std_fp16);
    if (inv_std < 1e-10f) inv_std = sqrtf((float) dim);
    const float scale = 1.0f / inv_std;

    /* Precomputed per-block dequant LUT (16 entries, scaled) */
    float lut[16];
    for (int c = 0; c < 16; c++) {
        lut[c] = turbo_kv_4b_codebook[c] * scale;
    }

    /* Unpack + codebook lookup into rotated[] */
    float rotated[TURBO_KV_BLOCK_SIZE];
    for (int i = 0; i < dim; i++) {
        const int byte_idx = i / 2;
        const int idx = (i & 1) ? (block->mse_indices[byte_idx] >> 4)
                                : (block->mse_indices[byte_idx] & 0x0F);
        rotated[i] = lut[idx];
    }
    for (int i = dim; i < TURBO_KV_BLOCK_SIZE; i++) {
        rotated[i] = 0.0f;
    }

    /* Inverse RHT */
    turbo_kv_rht_inverse(rotated, dim, TURBO_KV_DEFAULT_SEED);

    /* Scale by original norm */
    for (int i = 0; i < dim; i++) {
        dst[i] = rotated[i] * norm;
    }
}

void dequantize_row_turbo_kv_4b(const block_turbo_kv_4b * x, float * y, int64_t k) {
    GGML_ASSERT(k % TURBO_KV_BLOCK_SIZE == 0);
    const int64_t nb = k / TURBO_KV_BLOCK_SIZE;
    for (int64_t b = 0; b < nb; b++) {
        dequantize_block_turbo_kv_4b(x + b, y + b * TURBO_KV_BLOCK_SIZE, TURBO_KV_BLOCK_SIZE);
    }
}

/* Forward declare the per-block dot helper (defined below, after rotate_query). */
static inline float turbo_kv_4b_single_block_dot(
    const block_turbo_kv_4b * block, const float * q_rot_block, int dim_in_block);

/* ============================================================
 * Batched attention: rotate Q once, loop over all K positions.
 *
 * Mirrors tq_kv_1b_attention_multi (ggml-turbo-quant.c:377). The FA path
 * in ops.cpp calls this once per thread with a slice of K. The non-FA
 * vec_dot wrapper calls it with valid_count=1 as a single-position
 * fallback.
 * ============================================================ */
void turbo_kv_4b_attention_multi(
    const float              * query,
    const block_turbo_kv_4b  * kv_cache,
    float                    * scores,
    int                        valid_count,
    int                        head_dim,
    int                        k_stride_blocks)
{
    const int n_blocks = head_dim / TURBO_KV_BLOCK_SIZE;
    if (n_blocks * TURBO_KV_BLOCK_SIZE != head_dim || n_blocks <= 0 || n_blocks > 4) {
        /* Unsupported head_dim; fall back to per-call vec_dot. */
        for (int s = 0; s < valid_count; s++) {
            const block_turbo_kv_4b * blocks = &kv_cache[s * (k_stride_blocks > 0 ? k_stride_blocks : n_blocks)];
            ggml_vec_dot_turbo_kv_4b_f32(
                head_dim, &scores[s], 0, blocks, 0, query, 0, 1);
        }
        return;
    }

    if (k_stride_blocks <= 0) k_stride_blocks = n_blocks;

    /* Step 1: Pre-rotate query ONCE (per-block RHT). */
    float q_rot[4 * TURBO_KV_BLOCK_SIZE];
    turbo_kv_rotate_query(query, q_rot, head_dim);

    /* Step 2: Loop over K positions — sum per-block dot products. */
    for (int s = 0; s < valid_count; s++) {
        const block_turbo_kv_4b * k_row = &kv_cache[s * k_stride_blocks];
        float total = 0.0f;

        for (int b = 0; b < n_blocks; b++) {
            total += turbo_kv_4b_single_block_dot(
                &k_row[b], q_rot + b * TURBO_KV_BLOCK_SIZE, TURBO_KV_BLOCK_SIZE);
        }

        scores[s] = total;
    }
}

/* ============================================================
 * Query pre-rotation (public helper)
 *
 * Applies a per-block RHT(q) into out. dim may exceed TURBO_KV_BLOCK_SIZE
 * (e.g. Qwen3.5 uses head_dim=256 which is 2 × 128 blocks). Each
 * TURBO_KV_BLOCK_SIZE-aligned chunk is rotated independently with the
 * same seed, matching how the K cache stores multi-block rows. The full
 * dot product is then the sum of per-block dots (see vec_dot below).
 *
 * The output buffer must have room for nb × TURBO_KV_BLOCK_SIZE floats
 * where nb = ceil(dim / TURBO_KV_BLOCK_SIZE). Any trailing slack in the
 * last block is zero-padded before rotation.
 * ============================================================ */
void turbo_kv_rotate_query(const float * q, float * out, int dim) {
    if (dim <= 0) return;

    /* Whole blocks */
    int d = 0;
    while (d + TURBO_KV_BLOCK_SIZE <= dim) {
        memcpy(out + d, q + d, (size_t) TURBO_KV_BLOCK_SIZE * sizeof(float));
        turbo_kv_rht_forward(out + d, TURBO_KV_BLOCK_SIZE, TURBO_KV_DEFAULT_SEED);
        d += TURBO_KV_BLOCK_SIZE;
    }

    /* Tail block (zero-padded) */
    if (d < dim) {
        memcpy(out + d, q + d, (size_t) (dim - d) * sizeof(float));
        for (int i = dim; i < d + TURBO_KV_BLOCK_SIZE; i++) {
            out[i] = 0.0f;
        }
        turbo_kv_rht_forward(out + d, TURBO_KV_BLOCK_SIZE, TURBO_KV_DEFAULT_SEED);
    }
}

/* ============================================================
 * Scalar reference vec_dot: <f32 query, turbo_kv_4b row of blocks>
 *
 * The query MUST already be RHT-rotated per-block by turbo_kv_rotate_query.
 * The K row is n elements total; nb = ceil(n / 128) turbo_kv_4b blocks
 * stored contiguously. Each block has its own norm and inv_std. Because
 * the RHT is per-block AND orthogonal, the full dot product is the sum
 * of per-block dots:
 *
 *     <q, k>  ==  sum_b  norm_b * <q_rot_b, dequant_rot_b(k_b)>
 *
 * where q_rot_b is q_rot[b*128 .. b*128+127] and k_b is block b of the
 * K row. Mathematically correct because both RHT_b and the block-norm
 * factoring are linear operations, and RHT preserves dot products
 * within each block.
 *
 * n is the head dimension (e.g. 256 for Qwen3.5 attention heads, which
 * maps to 2 blocks per row).
 * ============================================================ */
static inline float turbo_kv_4b_single_block_dot(
    const block_turbo_kv_4b * block,
    const float * q_rot_block,
    int dim_in_block)
{
    const float norm = turbo_kv_fp16_to_fp32(block->norm);
    float inv_std = turbo_kv_fp16_to_fp32(block->inv_std_fp16);
    if (inv_std < 1e-10f) inv_std = sqrtf((float) dim_in_block);
    const float scale = 1.0f / inv_std;

    /* Pre-scaled 16-entry LUT (fused dequant + per-block scale) */
    float lut[16];
    for (int c = 0; c < 16; c++) {
        lut[c] = turbo_kv_4b_codebook[c] * scale;
    }

    const uint8_t * mi = block->mse_indices;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int d = 0;
    for (; d + 7 < dim_in_block; d += 8) {
        const uint8_t b0 = mi[(d + 0) / 2];
        const uint8_t b1 = mi[(d + 2) / 2];
        const uint8_t b2 = mi[(d + 4) / 2];
        const uint8_t b3 = mi[(d + 6) / 2];
        a0 += q_rot_block[d + 0] * lut[b0 & 0x0F];
        a1 += q_rot_block[d + 1] * lut[b0 >> 4];
        a2 += q_rot_block[d + 2] * lut[b1 & 0x0F];
        a3 += q_rot_block[d + 3] * lut[b1 >> 4];
        a0 += q_rot_block[d + 4] * lut[b2 & 0x0F];
        a1 += q_rot_block[d + 5] * lut[b2 >> 4];
        a2 += q_rot_block[d + 6] * lut[b3 & 0x0F];
        a3 += q_rot_block[d + 7] * lut[b3 >> 4];
    }
    float mse_dot = (a0 + a1) + (a2 + a3);
    for (; d < dim_in_block; d++) {
        const uint8_t bv = mi[d / 2];
        const int idx = (d & 1) ? (bv >> 4) : (bv & 0x0F);
        mse_dot += q_rot_block[d] * lut[idx];
    }

    return norm * mse_dot;
}

void ggml_vec_dot_turbo_kv_4b_f32(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by,
    int nrc)
{
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    /* vy is raw f32 from ggml. We rotate per call since the current
     * TURBO_KV_4B vec_dot_type is F32 (hoisting via TURBO_KV_Q_ROT_F32
     * was tried but regressed — see the comment in ggml-cpu.c on the
     * CPU trait registration). */
    const float * q_raw = (const float *) vy;
    const block_turbo_kv_4b * blocks = (const block_turbo_kv_4b *) vx;

    float q_rot[4 * TURBO_KV_BLOCK_SIZE];
    int rot_n = n;
    if (rot_n > 4 * TURBO_KV_BLOCK_SIZE) {
        rot_n = 4 * TURBO_KV_BLOCK_SIZE;
    }
    turbo_kv_rotate_query(q_raw, q_rot, rot_n);

    float total = 0.0f;
    int d = 0;
    int b = 0;
    while (d + TURBO_KV_BLOCK_SIZE <= rot_n) {
        total += turbo_kv_4b_single_block_dot(
            &blocks[b], q_rot + d, TURBO_KV_BLOCK_SIZE);
        d += TURBO_KV_BLOCK_SIZE;
        b++;
    }
    /* Tail block (partial) */
    if (d < rot_n) {
        total += turbo_kv_4b_single_block_dot(
            &blocks[b], q_rot + d, rot_n - d);
    }

    *s = total;
}

/* turbo_kv_precompute_rope_cache and turbo_kv_4b_attention_fused_rope
 * removed — superseded by split K cache (--cache-type-k q8_0:q4_0). */

/* ============================================================
 * TURBO_*B — RHT + Lloyd-Max weight quantization (2/3/4/5-bit)
 *
 * Generic implementation parametric by: block_size, bits, codebook, cent_max.
 * Block layout: [norm:2][inv_std:2][qs[block_size*bits/8]]
 *
 * Bit-packing uses a universal bit-stream approach:
 *   bit_offset = i * bits
 *   byte_idx   = bit_offset / 8
 *   bit_shift  = bit_offset % 8
 * This handles 2/3/4/5 bits per element without special-casing.
 * ============================================================ */

/* Bit-stream packing: write `bits`-wide index at element position `i` */
static inline void turbo_pack_bits(uint8_t * qs, int i, int idx, int bits) {
    const int bit_offset = i * bits;
    const int byte_idx   = bit_offset >> 3;
    const int bit_shift  = bit_offset & 7;
    /* Write may span two bytes when shift + bits > 8 */
    qs[byte_idx] |= (uint8_t)((idx & ((1 << bits) - 1)) << bit_shift);
    if (bit_shift + bits > 8) {
        qs[byte_idx + 1] |= (uint8_t)(idx >> (8 - bit_shift));
    }
}

/* Bit-stream unpacking: read `bits`-wide index at element position `i` */
static inline int turbo_unpack_bits(const uint8_t * qs, int i, int bits) {
    const int bit_offset = i * bits;
    const int byte_idx   = bit_offset >> 3;
    const int bit_shift  = bit_offset & 7;
    const int mask        = (1 << bits) - 1;
    int val = (qs[byte_idx] >> bit_shift);
    if (bit_shift + bits > 8) {
        val |= (qs[byte_idx + 1] << (8 - bit_shift));
    }
    return val & mask;
}

/* E8P CENT_MAX: max per-coord after sign/shift = 3.5 + 0.25 = 3.75.
 * We scale post-RHT values so max_abs maps to ~3.0 to stay inside the
 * dense part of the lattice. */
#define E8P_CENT_MAX 3.0f

/* Generic quantize: any bit-width, any codebook.
 * Special case: bits==2 uses E8P lattice VQ on groups of 8 elements
 * (scalar 4-level is too coarse at 2 bpw — catastrophic PPL). */
static void quantize_block_turbo(
    const float * src, uint8_t * dst, int block_size,
    const float * codebook, int n_levels, int bits, float cent_max)
{
    uint16_t * p_norm    = (uint16_t *)(dst + 0);
    uint16_t * p_inv_std = (uint16_t *)(dst + 2);
    uint8_t  * qs        = dst + 4;

    /* Step 1: L2 norm */
    float norm_sq = 0.0f;
    for (int i = 0; i < block_size; i++) norm_sq += src[i] * src[i];
    const float norm = sqrtf(norm_sq);
    *p_norm = turbo_kv_fp32_to_fp16(norm);

    /* Step 2: normalize to unit vector */
    float rotated[256];
    const float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    for (int i = 0; i < block_size; i++) rotated[i] = src[i] * inv_norm;

    /* Step 3: forward RHT */
    turbo_kv_rht_forward(rotated, block_size, TURBO_KV_DEFAULT_SEED);

    /* Step 4: compute max_abs and inv_std scale */
    float max_abs = 0.0f;
    for (int i = 0; i < block_size; i++) {
        float a = fabsf(rotated[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-10f) max_abs = 1.0f;

    if (bits == 2) {
        /* --- E8P lattice VQ path for TURBO_2B --- */
        const float inv_std = E8P_CENT_MAX / max_abs;
        *p_inv_std = turbo_kv_fp32_to_fp16(inv_std);

        /* 128 elements / 8 per group = 16 groups × 16-bit code = 32 bytes */
        GGML_ASSERT(block_size % 8 == 0);
        const int n_groups = block_size / 8;
        memset(qs, 0, 2 * n_groups);
        for (int g = 0; g < n_groups; g++) {
            float x[8];
            for (int j = 0; j < 8; j++) x[j] = rotated[g * 8 + j] * inv_std;
            uint16_t code = e8p_encode_16bit(x);
            qs[g * 2 + 0] = (uint8_t)(code & 0xFF);
            qs[g * 2 + 1] = (uint8_t)(code >> 8);
        }
        return;
    }

    /* --- Scalar Lloyd-Max path for bits >= 3 --- */
    const float inv_std = cent_max / max_abs;
    *p_inv_std = turbo_kv_fp32_to_fp16(inv_std);

    /* Step 5: nearest-centroid quantization + bit-stream packing */
    const int qs_bytes = (block_size * bits + 7) / 8;
    memset(qs, 0, qs_bytes);
    for (int i = 0; i < block_size; i++) {
        const float x = rotated[i] * inv_std;
        int best = 0;
        float best_dist = fabsf(x - codebook[0]);
        for (int c = 1; c < n_levels; c++) {
            float d = fabsf(x - codebook[c]);
            if (d < best_dist) { best_dist = d; best = c; }
        }
        turbo_pack_bits(qs, i, best, bits);
    }
}

/* Generic dequantize: any bit-width, any codebook.
 * Special case: bits==2 uses E8P lattice decode. */
static void dequantize_block_turbo(
    const uint8_t * src, float * dst, int block_size,
    const float * codebook, int n_levels, int bits)
{
    const float norm = turbo_kv_fp16_to_fp32(*(const uint16_t *)(src + 0));
    float inv_std = turbo_kv_fp16_to_fp32(*(const uint16_t *)(src + 2));
    if (inv_std < 1e-10f) inv_std = sqrtf((float)block_size);
    const float scale = 1.0f / inv_std;
    const uint8_t * qs = src + 4;

    float rotated[256];

    if (bits == 2) {
        /* E8P lattice decode: 16-bit code per group of 8 elements */
        const int n_groups = block_size / 8;
        for (int g = 0; g < n_groups; g++) {
            uint16_t code = (uint16_t)qs[g * 2 + 0] | ((uint16_t)qs[g * 2 + 1] << 8);
            float vals[8];
            e8p_decode_16bit(code, vals);
            for (int j = 0; j < 8; j++) rotated[g * 8 + j] = vals[j] * scale;
        }
    } else {
        /* Scalar codebook decode */
        float lut[32]; /* max 5-bit = 32 entries */
        for (int c = 0; c < n_levels; c++) lut[c] = codebook[c] * scale;
        for (int i = 0; i < block_size; i++) {
            int idx = turbo_unpack_bits(qs, i, bits);
            rotated[i] = lut[idx];
        }
    }

    /* Inverse RHT */
    turbo_kv_rht_inverse(rotated, block_size, TURBO_KV_DEFAULT_SEED);

    /* Scale by original norm */
    for (int i = 0; i < block_size; i++) dst[i] = rotated[i] * norm;
}

/* --- Codebook/bits config per type --- */

typedef struct {
    const float * codebook;
    int            n_levels;
    int            bits;
    float          cent_max;
    int            block_size;
    int            block_bytes;
} turbo_config_t;

static const turbo_config_t TURBO_2B_CONFIG = {
    NULL, 4, 2, TURBO_2B_CENT_MAX, TURBO_2B_BLOCK_SIZE, TURBO_2B_BYTES };
static const turbo_config_t TURBO_3B_CONFIG = {
    NULL, 8, 3, TURBO_3B_CENT_MAX, TURBO_3B_BLOCK_SIZE, TURBO_3B_BYTES };
static const turbo_config_t TURBO_4B_CONFIG = {
    NULL, 16, 4, TURBO_KV_4B_CENT_MAX, TURBO_4B_BLOCK_SIZE, TURBO_4B_BYTES };
static const turbo_config_t TURBO_4B_S_CONFIG = {
    NULL, 16, 4, TURBO_KV_4B_CENT_MAX, TURBO_4B_S_BLOCK_SIZE, TURBO_4B_S_BYTES };
static const turbo_config_t TURBO_5B_CONFIG = {
    NULL, 32, 5, TURBO_5B_CENT_MAX, TURBO_5B_BLOCK_SIZE, TURBO_5B_BYTES };

/* Per-bitrate codebook overrides (bits 2..5 → index 0..3). Used by both
   quantize (to pick centroids during index assignment) and dequant/vec_dot
   (to reconstruct values from stored indices). NULL means "use published
   default". Centroids are copied into a process-lifetime static buffer so
   callers don't need to keep their storage alive — the loader would
   otherwise risk the codebook vector going out of scope before inference
   dispatches run. Custom codebooks MUST be in [-cent_max, cent_max] range
   (same convention as the published Gaussian tables); the turbo-codebook
   tool rescales its Lloyd-Max output before writing to satisfy this. */
static float         g_turbo_override_storage[4][32];
static const float * g_turbo_quantize_override[4] = { NULL, NULL, NULL, NULL };

void turbo_set_quantize_codebook(int bits, const float * centroids) {
    if (bits < 2 || bits > 5) return;
    const int idx = bits - 2;
    const int n = 1 << bits;
    if (centroids) {
        for (int i = 0; i < n; i++) {
            g_turbo_override_storage[idx][i] = centroids[i];
        }
        g_turbo_quantize_override[idx] = g_turbo_override_storage[idx];
    } else {
        g_turbo_quantize_override[idx] = NULL;
    }
}

/* Resolve codebook pointer (can't use array address in static initializer) */
static const float * turbo_get_codebook(const turbo_config_t * cfg) {
    if (cfg->bits >= 2 && cfg->bits <= 5) {
        const float * ov = g_turbo_quantize_override[cfg->bits - 2];
        if (ov) return ov;
    }
    switch (cfg->bits) {
        case 2: return turbo_codebook_2bit;
        case 3: return turbo_codebook_3bit;
        case 4: return turbo_kv_4b_codebook;
        case 5: return turbo_codebook_5bit;
        default: return turbo_kv_4b_codebook;
    }
}

/* --- Generic row-wise quantize/dequant --- */

static void quantize_row_turbo_ref(const float * x, void * y, int64_t k,
                                    const turbo_config_t * cfg) {
    const float * cb = turbo_get_codebook(cfg);
    GGML_ASSERT(k % cfg->block_size == 0);
    const int64_t nb = k / cfg->block_size;
    for (int64_t b = 0; b < nb; b++) {
        quantize_block_turbo(
            x + b * cfg->block_size,
            (uint8_t *)y + b * cfg->block_bytes,
            cfg->block_size, cb, cfg->n_levels, cfg->bits, cfg->cent_max);
    }
}

static void dequantize_row_turbo(const void * x, float * y, int64_t k,
                                  const turbo_config_t * cfg) {
    const float * cb = turbo_get_codebook(cfg);
    GGML_ASSERT(k % cfg->block_size == 0);
    const int64_t nb = k / cfg->block_size;
    for (int64_t b = 0; b < nb; b++) {
        dequantize_block_turbo(
            (const uint8_t *)x + b * cfg->block_bytes,
            y + b * cfg->block_size,
            cfg->block_size, cb, cfg->n_levels, cfg->bits);
    }
}

/* --- Public API: one-line wrappers per type --- */

#define TURBO_QUANT_ROW_REF(suffix, cfg) \
void quantize_row_turbo_##suffix##_ref(const float * GGML_RESTRICT x, \
    block_turbo_##suffix * GGML_RESTRICT y, int64_t k) { \
    quantize_row_turbo_ref(x, y, k, &cfg); \
}

#define TURBO_DEQUANT_ROW(suffix, cfg) \
void dequantize_row_turbo_##suffix(const block_turbo_##suffix * GGML_RESTRICT x, \
    float * GGML_RESTRICT y, int64_t k) { \
    dequantize_row_turbo(x, y, k, &cfg); \
}

TURBO_QUANT_ROW_REF(2b,   TURBO_2B_CONFIG)
TURBO_DEQUANT_ROW  (2b,   TURBO_2B_CONFIG)
TURBO_QUANT_ROW_REF(3b,   TURBO_3B_CONFIG)
TURBO_DEQUANT_ROW  (3b,   TURBO_3B_CONFIG)
TURBO_QUANT_ROW_REF(4b,   TURBO_4B_CONFIG)
TURBO_DEQUANT_ROW  (4b,   TURBO_4B_CONFIG)

TURBO_QUANT_ROW_REF(4b_s, TURBO_4B_S_CONFIG)
TURBO_DEQUANT_ROW  (4b_s, TURBO_4B_S_CONFIG)
TURBO_QUANT_ROW_REF(5b,   TURBO_5B_CONFIG)
TURBO_DEQUANT_ROW  (5b,   TURBO_5B_CONFIG)

/* --- imatrix-weighted quantization (generic) --- */

static void quantize_block_turbo_weighted(
    const float * src, uint8_t * dst, int block_size,
    const float * codebook, int n_levels, int bits, float cent_max,
    const float * weights)
{
    /* Weights are currently ignored: per-element nearest-centroid is invariant
     * to a per-element positive scalar weight (doesn't change argmin_c |x-c|).
     * The only ways imatrix could improve per-block quantization are scale
     * selection (tried 5-candidate search, regresses 4B) or mixed precision
     * (needs bit-width variance). Until one of those is re-introduced, the
     * weighted path must produce bits identical to the unweighted path — in
     * particular it must take the E8P lattice branch for bits==2. Delegating
     * to quantize_block_turbo guarantees this by construction. */
    (void)weights;
    quantize_block_turbo(src, dst, block_size, codebook, n_levels, bits, cent_max);
}

/* --- imatrix-aware ggml_quantize_chunk wrappers (generic) --- */

static size_t quantize_turbo_generic(
    const float * src, void * dst, int64_t nrows, int64_t n_per_row,
    const float * quant_weights, const turbo_config_t * cfg, enum ggml_type type)
{
    const size_t row_size = ggml_row_size(type, n_per_row);
    const float * cb = turbo_get_codebook(cfg);
    if (!quant_weights) {
        quantize_row_turbo_ref(src, dst, nrows * n_per_row, cfg);
    } else {
        char * qrow = (char *)dst;
        for (int64_t row = 0; row < nrows; row++) {
            GGML_ASSERT(n_per_row % cfg->block_size == 0);
            const int64_t nb = n_per_row / cfg->block_size;
            for (int64_t b = 0; b < nb; b++) {
                quantize_block_turbo_weighted(
                    src + b * cfg->block_size,
                    (uint8_t *)qrow + b * cfg->block_bytes,
                    cfg->block_size, cb, cfg->n_levels, cfg->bits, cfg->cent_max,
                    quant_weights + b * cfg->block_size);
            }
            src += n_per_row;
            qrow += row_size;
        }
    }
    return nrows * row_size;
}

size_t quantize_turbo_2b(const float * src, void * dst,
    int64_t nrows, int64_t n_per_row, const float * qw) {
    return quantize_turbo_generic(src, dst, nrows, n_per_row, qw, &TURBO_2B_CONFIG, GGML_TYPE_TURBO_2B);
}
size_t quantize_turbo_3b(const float * src, void * dst,
    int64_t nrows, int64_t n_per_row, const float * qw) {
    return quantize_turbo_generic(src, dst, nrows, n_per_row, qw, &TURBO_3B_CONFIG, GGML_TYPE_TURBO_3B);
}
size_t quantize_turbo_4b(const float * src, void * dst,
    int64_t nrows, int64_t n_per_row, const float * qw) {
    return quantize_turbo_generic(src, dst, nrows, n_per_row, qw, &TURBO_4B_CONFIG, GGML_TYPE_TURBO_4B);
}
size_t quantize_turbo_4b_s(const float * src, void * dst,
    int64_t nrows, int64_t n_per_row, const float * qw) {
    return quantize_turbo_generic(src, dst, nrows, n_per_row, qw, &TURBO_4B_S_CONFIG, GGML_TYPE_TURBO_4B_S);
}
size_t quantize_turbo_5b(const float * src, void * dst,
    int64_t nrows, int64_t n_per_row, const float * qw) {
    return quantize_turbo_generic(src, dst, nrows, n_per_row, qw, &TURBO_5B_CONFIG, GGML_TYPE_TURBO_5B);
}

/* --- Generic vec_dot: dequant-then-dot --- */

static float turbo_block_dot_dequant(
    const uint8_t * block_data, const float * activation,
    const turbo_config_t * cfg)
{
    const float * cb = turbo_get_codebook(cfg);
    const float norm = turbo_kv_fp16_to_fp32(*(const uint16_t *)(block_data + 0));
    float inv_std = turbo_kv_fp16_to_fp32(*(const uint16_t *)(block_data + 2));
    if (inv_std < 1e-10f) inv_std = sqrtf((float)cfg->block_size);
    const float scale = 1.0f / inv_std;
    const uint8_t * qs = block_data + 4;

    float rotated[256];

    if (cfg->bits == 2) {
        /* E8P lattice decode */
        const int n_groups = cfg->block_size / 8;
        for (int g = 0; g < n_groups; g++) {
            uint16_t code = (uint16_t)qs[g * 2 + 0] | ((uint16_t)qs[g * 2 + 1] << 8);
            float vals[8];
            e8p_decode_16bit(code, vals);
            for (int j = 0; j < 8; j++) rotated[g * 8 + j] = vals[j] * scale;
        }
    } else {
        float lut[32];
        for (int c = 0; c < cfg->n_levels; c++) lut[c] = cb[c] * scale;
        for (int i = 0; i < cfg->block_size; i++) {
            int idx = turbo_unpack_bits(qs, i, cfg->bits);
            rotated[i] = lut[idx];
        }
    }

    turbo_kv_rht_inverse(rotated, cfg->block_size, TURBO_KV_DEFAULT_SEED);

    float acc = 0.0f;
    for (int i = 0; i < cfg->block_size; i++) acc += rotated[i] * activation[i];
    return acc * norm;
}

/* Fused RHT-space dot: pre-rotate activation, dot without inverse RHT.
 * <w, x> = <RHT^{-1}(w_rot), x> = <w_rot, RHT(x)> because RHT is orthogonal.
 * Eliminates the inverse FWHT per block (7-stage butterfly × n_blocks). */
static float turbo_block_dot_fused(
    const uint8_t * block_data, const float * act_rotated,
    const turbo_config_t * cfg)
{
    const float * cb = turbo_get_codebook(cfg);
    const float norm = turbo_kv_fp16_to_fp32(*(const uint16_t *)(block_data + 0));
    float inv_std = turbo_kv_fp16_to_fp32(*(const uint16_t *)(block_data + 2));
    if (inv_std < 1e-10f) inv_std = sqrtf((float)cfg->block_size);
    const float scale = 1.0f / inv_std;
    const uint8_t * qs = block_data + 4;

    float acc = 0.0f;

    if (cfg->bits == 2) {
        /* E8P: decode in RHT space, dot with rotated activation */
        const int n_groups = cfg->block_size / 8;
        for (int g = 0; g < n_groups; g++) {
            uint16_t code = (uint16_t)qs[g * 2 + 0] | ((uint16_t)qs[g * 2 + 1] << 8);
            float vals[8];
            e8p_decode_16bit(code, vals);
            for (int j = 0; j < 8; j++) {
                acc += (vals[j] * scale) * act_rotated[g * 8 + j];
            }
        }
    } else {
        for (int i = 0; i < cfg->block_size; i++) {
            int idx = turbo_unpack_bits(qs, i, cfg->bits);
            acc += (cb[idx] * scale) * act_rotated[i];
        }
    }
    return acc * norm;
}

static void ggml_vec_dot_turbo_generic(
    int n, float * GGML_RESTRICT s,
    const void * GGML_RESTRICT vx,
    const void * GGML_RESTRICT vy,
    const turbo_config_t * cfg)
{
    const uint8_t * wd = (const uint8_t *)vx;
    const float   * ac = (const float *)vy;

    /* Weight vec_dot can be called with very long rows (n=4096+) during CPU
     * inference. Use dequant-then-dot path to avoid stack buffer limits. */
    float total = 0.0f;
    int d = 0, offset = 0;
    while (d + cfg->block_size <= n) {
        total += turbo_block_dot_dequant(wd + offset, ac + d, cfg);
        d += cfg->block_size;
        offset += cfg->block_bytes;
    }
    *s = total;
}

#define TURBO_VEC_DOT(suffix, cfg) \
void ggml_vec_dot_turbo_##suffix##_f32( \
    int n, float * GGML_RESTRICT s, size_t bs, \
    const void * GGML_RESTRICT vx, size_t bx, \
    const void * GGML_RESTRICT vy, size_t by, int nrc) { \
    GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(nrc); \
    ggml_vec_dot_turbo_generic(n, s, vx, vy, &cfg); \
}

TURBO_VEC_DOT(2b,   TURBO_2B_CONFIG)
TURBO_VEC_DOT(3b,   TURBO_3B_CONFIG)
TURBO_VEC_DOT(4b,   TURBO_4B_CONFIG)
TURBO_VEC_DOT(4b_s, TURBO_4B_S_CONFIG)
TURBO_VEC_DOT(5b,   TURBO_5B_CONFIG)
