/**
 * ggml-turbo-quant.c -- quant.cpp 1-bit KV cache quantization for llama.cpp
 *
 * Apache 2.0 License, QuantumAI Inc.
 *
 * Self-contained C99 implementation. No external dependencies beyond libc/libm.
 * Ported from quant.cpp src/core/tq_rht.c and src/core/tq_turbo_kv.c.
 *
 * Algorithm overview:
 *   Quantize:   L2-normalize -> RHT (random signs + Walsh-Hadamard) -> sign bits
 *   Dequantize: signs -> scale(sqrt(2/pi)/sqrt(dim)) -> inverse RHT -> scale(norm)
 *   Attention:  RHT(query) -> sign bits -> XOR + popcount -> Hamming score
 */

#include "ggml-turbo-quant.h"
#include <math.h>
#include <string.h>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif
#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif
#if defined(__SSE4_2__) || defined(__POPCNT__)
#include <nmmintrin.h>
#endif

/* ============================================================
 * FP16 <-> FP32 conversion (self-contained, no ggml dependency)
 * ============================================================ */

static uint16_t tq_fp32_to_fp16(float v) {
    union { float f; uint32_t u; } bits;
    bits.f = v;
    uint32_t sign = (bits.u >> 16) & 0x8000;
    int32_t  exp  = ((bits.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits.u >> 13) & 0x03FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float tq_fp16_to_fp32(uint16_t h) {
    union { float f; uint32_t u; } bits;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0)  { bits.u = sign; return bits.f; }
    if (exp == 31) { bits.u = sign | 0x7F800000 | (mant << 13); return bits.f; }
    exp = exp - 15 + 127;
    bits.u = sign | (exp << 23) | (mant << 13);
    return bits.f;
}

/* ============================================================
 * Constants
 * ============================================================ */

#define TQ_PI   3.14159265358979323846f
#define TQ_PI_2 1.5707963267948966f     /* pi/2 */

/* Default RHT seed -- all blocks use the same seed so we can
 * pre-rotate the query once and reuse across all keys. */
#define TQ_DEFAULT_SEED 0x12345678u

/* ============================================================
 * Random Hadamard Transform (RHT) -- self-contained port
 *
 * RHT = (1/sqrt(n)) * H * D where:
 *   D = diagonal random sign matrix (from seed)
 *   H = Walsh-Hadamard butterfly transform
 *
 * Properties:
 *   - RHT is orthogonal: preserves inner products
 *   - O(n log n) computation, no matrix storage
 *   - Decorrelates channels, making scalar quantization near-optimal
 *   - Self-inverse (up to scaling): H * H = n * I
 * ============================================================ */

/* Deterministic random sign from seed + index (Knuth multiplicative hash) */
static int tq_random_sign(uint32_t seed, int idx) {
    uint32_t h = seed ^ (uint32_t)idx;
    h = h * 2654435761u;
    return (h & 1) ? 1 : -1;
}

/* In-place Walsh-Hadamard Transform: O(n log n) butterfly.
 * n must be a power of 2. Self-inverse up to scaling: WHT(WHT(x)) = n * x. */
static void tq_walsh_hadamard(float * data, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = data[i + j];
                float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
}

/* Forward RHT: random sign flip -> WHT -> normalize by 1/sqrt(n) */
static void tq_rht_forward(float * data, int n, uint32_t seed) {
    if (!data || n <= 0) return;

    /* Round down to nearest power of 2 */
    int n2 = 1;
    while (n2 * 2 <= n) n2 *= 2;

    /* Step 1: Random sign flip (diagonal D matrix) */
    for (int i = 0; i < n2; i++) {
        data[i] *= (float)tq_random_sign(seed, i);
    }

    /* Step 2: Walsh-Hadamard butterfly */
    tq_walsh_hadamard(data, n2);

    /* Step 3: Normalize by 1/sqrt(n) for orthogonal transform */
    float scale = 1.0f / sqrtf((float)n2);
    for (int i = 0; i < n2; i++) {
        data[i] *= scale;
    }
}

/* Inverse RHT: normalize -> WHT -> same sign flip.
 * Since H is self-inverse up to scaling and D*D = I:
 *   RHT     = (1/sqrt(n)) * H * D
 *   RHT^-1  = (1/sqrt(n)) * D * H  = D * (1/sqrt(n)) * H */
static void tq_rht_inverse(float * data, int n, uint32_t seed) {
    if (!data || n <= 0) return;

    int n2 = 1;
    while (n2 * 2 <= n) n2 *= 2;

    /* Step 1: Normalize by 1/sqrt(n) */
    float scale = 1.0f / sqrtf((float)n2);
    for (int i = 0; i < n2; i++) {
        data[i] *= scale;
    }

    /* Step 2: Walsh-Hadamard (self-inverse up to scaling) */
    tq_walsh_hadamard(data, n2);

    /* Step 3: Same random sign flip (D * D = I) */
    for (int i = 0; i < n2; i++) {
        data[i] *= (float)tq_random_sign(seed, i);
    }
}

/* ============================================================
 * Portable popcount (for platforms without __builtin_popcount)
 * ============================================================ */

static int tq_popcount8(uint8_t x) {
    int c = 0;
    while (x) { c++; x &= x - 1; }  /* Kernighan's bit trick */
    return c;
}

/* ============================================================
 * Quantize: float -> block_tq_kv_1b
 *
 * Pipeline per 128-element block:
 *   1. Compute L2 norm of block
 *   2. L2-normalize the block
 *   3. Apply RHT (random signs + Walsh-Hadamard + 1/sqrt(n))
 *   4. Extract sign bits (1 = positive, 0 = negative)
 *   5. Store norm (FP16) and RHT seed
 * ============================================================ */

void quantize_row_tq_kv_1b_ref(const float * x, block_tq_kv_1b * y, int64_t k) {
    const int block_size = TQ_KV_1B_BLOCK_SIZE;
    const int64_t num_blocks = k / block_size;

    for (int64_t b = 0; b < num_blocks; b++) {
        const float * src = x + b * block_size;
        block_tq_kv_1b * block = &y[b];

        /* Step 1: Compute L2 norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < block_size; i++) {
            norm_sq += src[i] * src[i];
        }
        float norm = sqrtf(norm_sq);
        block->norm = tq_fp32_to_fp16(norm);
        block->_pad = 0;

        /* Step 2: Normalize and copy to working buffer */
        float rotated[TQ_KV_1B_BLOCK_SIZE];
        float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
        for (int i = 0; i < block_size; i++) {
            rotated[i] = src[i] * inv_norm;
        }

        /* Step 3: Apply RHT */
        uint32_t seed = TQ_DEFAULT_SEED;
        block->rht_seed = seed;
        tq_rht_forward(rotated, block_size, seed);

        /* Step 4: Extract sign bits -- 1 bit per dimension, LSB-first */
        int sign_bytes = block_size / 8;
        memset(block->signs, 0, (size_t)sign_bytes);
        for (int i = 0; i < block_size; i++) {
            if (rotated[i] > 0.0f) {
                block->signs[i / 8] |= (uint8_t)(1 << (i % 8));
            }
        }
    }
}

/* ============================================================
 * Dequantize: block_tq_kv_1b -> float
 *
 * Pipeline per block:
 *   1. Reconstruct sign vector as +/- scale in rotated space
 *      (scale = sqrt(2/pi) / sqrt(dim), the expected absolute value
 *       of a half-normal distribution after RHT)
 *   2. Apply inverse RHT
 *   3. Scale by original L2 norm
 *
 * Note: This is a rough point-wise reconstruction. The real value
 * of 1-bit quantization is in Hamming attention (below).
 * ============================================================ */

void dequantize_row_tq_kv_1b(const block_tq_kv_1b * x, float * y, int64_t k) {
    const int block_size = TQ_KV_1B_BLOCK_SIZE;
    const int64_t num_blocks = k / block_size;

    for (int64_t b = 0; b < num_blocks; b++) {
        const block_tq_kv_1b * block = &x[b];
        float * dst = y + b * block_size;

        float norm = tq_fp16_to_fp32(block->norm);
        uint32_t seed = block->rht_seed;

        /* Reconstruct sign vector in rotated space.
         * After RHT, coordinates are ~N(0, 1/sqrt(dim)).
         * Expected |x| for half-normal = sqrt(2/pi) * sigma = sqrt(2/pi) / sqrt(dim). */
        float scale = sqrtf(2.0f / TQ_PI) / sqrtf((float)block_size);
        float rotated[TQ_KV_1B_BLOCK_SIZE];
        for (int i = 0; i < block_size; i++) {
            int bit = (block->signs[i / 8] >> (i % 8)) & 1;
            rotated[i] = bit ? scale : -scale;
        }

        /* Inverse RHT */
        tq_rht_inverse(rotated, block_size, seed);

        /* Scale by original norm */
        for (int i = 0; i < block_size; i++) {
            dst[i] = rotated[i] * norm;
        }
    }
}

/* ============================================================
 * Attention: XOR + popcount Hamming distance
 *
 * Ultra-fast attention using bitwise operations:
 *   1. RHT(query) computed ONCE (all keys share the same seed)
 *   2. Extract query sign bits ONCE
 *   3. Per key: XOR + popcount -> Hamming distance -> score
 *
 * Inner product estimator (from QJL/quant.cpp theory):
 *   <q, k> ~ q_norm * k_norm * sqrt(pi/2) / dim * (2*agree - dim)
 *
 * where agree = dim - hamming_distance(q_signs, k_signs).
 *
 * Theoretical cosine similarity: 2/pi ~ 0.637 for random vectors.
 * In practice, attention patterns are preserved because relative
 * ordering of scores is maintained (important tokens stay important).
 * ============================================================ */

void tq_kv_1b_attention(const float * query, const block_tq_kv_1b * kv_cache,
                         float * scores, int seq_len, int head_dim) {
    const int dim = (head_dim <= TQ_KV_1B_BLOCK_SIZE) ? head_dim : TQ_KV_1B_BLOCK_SIZE;

    float scale_factor = sqrtf(TQ_PI_2) / (float)dim;

    /* Step 1: RHT(query) computed ONCE.
     * Since all keys use TQ_DEFAULT_SEED, a single rotation suffices.
     * RHT is orthogonal: <q, Pi^T * k_rot> = <Pi*q, k_rot>. */
    float q_rot[TQ_KV_1B_BLOCK_SIZE];
    memcpy(q_rot, query, (size_t)dim * sizeof(float));
    {
        int i;
        for (i = dim; i < TQ_KV_1B_BLOCK_SIZE; i++) q_rot[i] = 0.0f;
    }
    tq_rht_forward(q_rot, dim, TQ_DEFAULT_SEED);

    /* Step 2: Compute query L2 norm */
    float q_norm_sq = 0.0f;
    {
        int i;
        for (i = 0; i < dim; i++) {
            q_norm_sq += query[i] * query[i];
        }
    }
    float q_norm = sqrtf(q_norm_sq);

    /* Step 3: Extract query sign bits */
    int sign_bytes = dim / 8;
    uint8_t q_signs[TQ_KV_1B_BLOCK_SIZE / 8];
    if (sign_bytes > 0) memset(q_signs, 0, (size_t)sign_bytes);
    {
        int i;
        for (i = 0; i < dim; i++) {
            if (q_rot[i] > 0.0f) {
                q_signs[i / 8] |= (uint8_t)(1 << (i % 8));
            }
        }
    }

    /* Step 4: Per-key Hamming attention */
    {
        int seq;
        for (seq = 0; seq < seq_len; seq++) {
            const block_tq_kv_1b * blk = &kv_cache[seq];
            float k_norm = tq_fp16_to_fp32(blk->norm);

            /* XOR + popcount -> Hamming distance */
            int hamming = 0;
            {
                int b;
                for (b = 0; b < sign_bytes; b++) {
                    uint8_t xor_byte = q_signs[b] ^ blk->signs[b];
                    hamming += tq_popcount8(xor_byte);
                }
            }

            int agree = dim - hamming;
            float score = q_norm * k_norm * scale_factor * (float)(2 * agree - dim);
            scores[seq] = score;
        }
    }
}

/* ============================================================
 * Multi-block Hamming attention with SSE4.2 (Westmere-friendly)
 *
 * For head_dim > 128, each K is stored as N = head_dim/128 blocks.
 * Score = sum over blocks of per-block Hamming-distance contribution.
 *
 * Key optimisation: query is rotated/sign-extracted ONCE per block
 * (per call), then per K position we do:
 *   N x (PXOR 16B + 2x POPCNT64 + ADD)
 *
 * Westmere has POPCNT (introduced with SSE4.2) and SSE4.1 PEXTRQ.
 * Each 16-byte block compares in ~6 cycles vs ~40 cycles scalar.
 * ============================================================ */

#if defined(__SSE4_2__) || defined(__POPCNT__)
static inline int tq_xor_popcount_16(const uint8_t * a, const uint8_t * b) {
    __m128i va = _mm_loadu_si128((const __m128i *) a);
    __m128i vb = _mm_loadu_si128((const __m128i *) b);
    __m128i vx = _mm_xor_si128(va, vb);
    /* _mm_extract_epi64 requires SSE4.1; popcnt64 requires POPCNT (SSE4.2). */
    long long lo = _mm_extract_epi64(vx, 0);
    long long hi = _mm_extract_epi64(vx, 1);
    return (int) (_mm_popcnt_u64((unsigned long long) lo) +
                  _mm_popcnt_u64((unsigned long long) hi));
}
#else
static inline int tq_xor_popcount_16(const uint8_t * a, const uint8_t * b) {
    int h = 0;
    for (int i = 0; i < 16; i++) {
        uint8_t x = a[i] ^ b[i];
        h += tq_popcount8(x);
    }
    return h;
}
#endif

#define TQ_MAX_BLOCKS_PER_KEY 4   /* head_dim up to 512 */

void tq_kv_1b_attention_multi(const float * query,
                               const block_tq_kv_1b * kv_cache,
                               float * scores, int seq_len, int head_dim) {
    const int n_blocks = head_dim / TQ_KV_1B_BLOCK_SIZE;
    /* If head_dim is not a multiple of the block size, fall back to the
     * single-block path which already handles dim<=128. */
    if (n_blocks * TQ_KV_1B_BLOCK_SIZE != head_dim || n_blocks > TQ_MAX_BLOCKS_PER_KEY) {
        tq_kv_1b_attention(query, kv_cache, scores, seq_len, head_dim);
        return;
    }

    /* Per-block precomputation: rotate query slice, sign-extract, compute norm. */
    uint8_t   q_signs[TQ_MAX_BLOCKS_PER_KEY][TQ_KV_1B_BLOCK_SIZE / 8];
    float     q_norm [TQ_MAX_BLOCKS_PER_KEY];
    const float per_block_scale = sqrtf(TQ_PI_2) / (float) TQ_KV_1B_BLOCK_SIZE;

    for (int b = 0; b < n_blocks; b++) {
        const float * q_slice = query + b * TQ_KV_1B_BLOCK_SIZE;

        float q_rot[TQ_KV_1B_BLOCK_SIZE];
        memcpy(q_rot, q_slice, TQ_KV_1B_BLOCK_SIZE * sizeof(float));
        tq_rht_forward(q_rot, TQ_KV_1B_BLOCK_SIZE, TQ_DEFAULT_SEED);

        float sumsq = 0.0f;
        for (int i = 0; i < TQ_KV_1B_BLOCK_SIZE; i++) sumsq += q_slice[i] * q_slice[i];
        q_norm[b] = sqrtf(sumsq);

        memset(q_signs[b], 0, TQ_KV_1B_BLOCK_SIZE / 8);
        for (int i = 0; i < TQ_KV_1B_BLOCK_SIZE; i++) {
            if (q_rot[i] > 0.0f) {
                q_signs[b][i >> 3] |= (uint8_t) (1u << (i & 7));
            }
        }
    }

    /* Per-K-position attention: sum n_blocks Hamming contributions. */
    for (int s = 0; s < seq_len; s++) {
        const block_tq_kv_1b * blocks = &kv_cache[s * n_blocks];
        float total = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            int hamming = tq_xor_popcount_16(q_signs[b], blocks[b].signs);
            int agree   = TQ_KV_1B_BLOCK_SIZE - hamming;
            float k_norm = tq_fp16_to_fp32(blocks[b].norm);
            total += q_norm[b] * k_norm * per_block_scale * (float) (2 * agree - TQ_KV_1B_BLOCK_SIZE);
        }
        scores[s] = total;
    }
}
