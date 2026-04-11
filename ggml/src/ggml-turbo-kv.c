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

/* Per-centroid decision boundaries are midpoints between consecutive
 * centroids. For nearest-centroid lookup we compute |x - c| and take min.
 * With only 16 entries, linear scan beats binary search (~15 cmps). */

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

/* Round n down to the nearest power of 2 (n2 <= n, n2 is a power of 2). */
static inline int turbo_kv_pow2_floor(int n) {
    int n2 = 1;
    while ((n2 << 1) <= n) n2 <<= 1;
    return n2;
}

void turbo_kv_rht_forward(float * data, int n, uint32_t seed) {
    if (!data || n <= 0) return;
    const int n2 = turbo_kv_pow2_floor(n);

    /* Step 1: random sign flip */
    for (int i = 0; i < n2; i++) {
        data[i] *= (float) turbo_kv_random_sign(seed, i);
    }

    /* Step 2: Walsh-Hadamard butterfly */
    walsh_hadamard(data, n2);

    /* Step 3: normalize by 1/sqrt(n2) */
    const float scale = 1.0f / sqrtf((float) n2);
    for (int i = 0; i < n2; i++) {
        data[i] *= scale;
    }
}

void turbo_kv_rht_inverse(float * data, int n, uint32_t seed) {
    if (!data || n <= 0) return;
    const int n2 = turbo_kv_pow2_floor(n);

    /* Inverse: scale, butterfly, sign-flip (reverse order of forward) */
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
static inline float turbo_kv_fp16_to_fp32(uint16_t h) {
    ggml_fp16_t h16;
    memcpy(&h16, &h, sizeof(h16));
    return GGML_FP16_TO_FP32(h16);
}

static inline uint16_t turbo_kv_fp32_to_fp16(float f) {
    const ggml_fp16_t r = GGML_FP32_TO_FP16(f);
    uint16_t out;
    memcpy(&out, &r, sizeof(out));
    return out;
}

/* ============================================================
 * Quantize one block (128 elements -> 72 bytes)
 * ============================================================ */
static void quantize_block_turbo_kv_4b(const float * src, block_turbo_kv_4b * block, int dim) {
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

    /* Step 2: normalize to unit vector */
    float rotated[TURBO_KV_BLOCK_SIZE];
    const float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    for (int i = 0; i < dim; i++) {
        rotated[i] = src[i] * inv_norm;
    }
    for (int i = dim; i < TURBO_KV_BLOCK_SIZE; i++) {
        rotated[i] = 0.0f;
    }

    /* Step 3: RHT (in-place) */
    turbo_kv_rht_forward(rotated, dim, TURBO_KV_DEFAULT_SEED);

    /* Step 4: compute max_abs and per-block inv_std */
    float max_abs = 0.0f;
    for (int i = 0; i < dim; i++) {
        const float a = fabsf(rotated[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-10f) max_abs = 1.0f;
    const float inv_std = TURBO_KV_4B_CENT_MAX / max_abs;
    block->inv_std_fp16 = turbo_kv_fp32_to_fp16(inv_std);

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

/* ============================================================
 * Query pre-rotation (public helper)
 *
 * Applies RHT(q) into out, zero-padded to TURBO_KV_BLOCK_SIZE. dim must
 * be <= BK. This is the expensive step at the Q boundary but it is only
 * called ONCE per attention row — the K batch shares the rotation.
 * ============================================================ */
void turbo_kv_rotate_query(const float * q, float * out, int dim) {
    if (dim > TURBO_KV_BLOCK_SIZE) dim = TURBO_KV_BLOCK_SIZE;
    memcpy(out, q, (size_t) dim * sizeof(float));
    for (int i = dim; i < TURBO_KV_BLOCK_SIZE; i++) {
        out[i] = 0.0f;
    }
    turbo_kv_rht_forward(out, dim, TURBO_KV_DEFAULT_SEED);
}

/* ============================================================
 * Scalar reference vec_dot: <f32 query, turbo_kv_4b block>
 *
 * The query MUST already be RHT-rotated and zero-padded to BK. In ggml's
 * type-traits API, kq_vec_dot passes the pre-quantized Q (already routed
 * through q_to_vec_dot) as vy. For TURBO_KV_4B, vec_dot_type = F32 and
 * q_to_vec_dot is our turbo_kv_rotate_query wrapper that runs once per
 * row and caches into the per-thread scratch buffer used by the outer FA
 * loop.
 *
 * n is the head dimension (e.g. 256 for Qwen3.5 attention heads).
 * ============================================================ */
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

    const float * q_rot = (const float *) vy;
    const block_turbo_kv_4b * block = (const block_turbo_kv_4b *) vx;

    int dim = n;
    if (dim > TURBO_KV_BLOCK_SIZE) dim = TURBO_KV_BLOCK_SIZE;

    const float norm = turbo_kv_fp16_to_fp32(block->norm);
    float inv_std = turbo_kv_fp16_to_fp32(block->inv_std_fp16);
    if (inv_std < 1e-10f) inv_std = sqrtf((float) dim);
    const float scale = 1.0f / inv_std;

    /* Pre-scaled 16-entry LUT (fused dequant + per-block scale) */
    float lut[16];
    for (int c = 0; c < 16; c++) {
        lut[c] = turbo_kv_4b_codebook[c] * scale;
    }

    /* Dot product in rotated space — RHT is orthogonal so
     * <q, k>  ==  <RHT(q), RHT(k)>.
     */
    const uint8_t * mi = block->mse_indices;
    float mse_dot = 0.0f;
    /* Unrolled by 8 elements per iteration for ILP */
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int d = 0;
    for (; d + 7 < dim; d += 8) {
        const uint8_t b0 = mi[(d + 0) / 2];
        const uint8_t b1 = mi[(d + 2) / 2];
        const uint8_t b2 = mi[(d + 4) / 2];
        const uint8_t b3 = mi[(d + 6) / 2];
        a0 += q_rot[d + 0] * lut[b0 & 0x0F];
        a1 += q_rot[d + 1] * lut[b0 >> 4];
        a2 += q_rot[d + 2] * lut[b1 & 0x0F];
        a3 += q_rot[d + 3] * lut[b1 >> 4];
        a0 += q_rot[d + 4] * lut[b2 & 0x0F];
        a1 += q_rot[d + 5] * lut[b2 >> 4];
        a2 += q_rot[d + 6] * lut[b3 & 0x0F];
        a3 += q_rot[d + 7] * lut[b3 >> 4];
    }
    mse_dot = (a0 + a1) + (a2 + a3);
    for (; d < dim; d++) {
        const uint8_t bv = mi[d / 2];
        const int idx = (d & 1) ? (bv >> 4) : (bv & 0x0F);
        mse_dot += q_rot[d] * lut[idx];
    }

    *s = norm * mse_dot;
}
