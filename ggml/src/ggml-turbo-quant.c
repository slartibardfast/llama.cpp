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
                               float * scores, int seq_len, int head_dim,
                               int k_stride_blocks) {
    const int n_blocks = head_dim / TQ_KV_1B_BLOCK_SIZE;
    /* If head_dim is not a multiple of the block size, fall back to the
     * single-block path which already handles dim<=128. */
    if (n_blocks * TQ_KV_1B_BLOCK_SIZE != head_dim || n_blocks > TQ_MAX_BLOCKS_PER_KEY) {
        tq_kv_1b_attention(query, kv_cache, scores, seq_len, head_dim);
        return;
    }

    /* k_stride_blocks: number of blocks between consecutive K positions
     * in the cache buffer. For a GQA layout with n_kv_heads packed per
     * row, this is n_embd_k_gqa/128 (all heads' blocks), not just
     * head_dim/128 (one head's blocks). Pass 0 for the legacy default. */
    if (k_stride_blocks <= 0) k_stride_blocks = n_blocks;

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

    /* Per-K-position attention: sum n_blocks Hamming contributions.
     * The stride between consecutive K positions may be larger than
     * n_blocks when the cache row contains multiple KV heads (GQA). */
    for (int s = 0; s < seq_len; s++) {
        const block_tq_kv_1b * blocks = &kv_cache[s * k_stride_blocks];
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

/* ============================================================
 * TurboQuant V 4-bit quantization
 *
 * Symmetric q4_0-style scheme, 128-element blocks (vs q4_0's 32).
 * Layout matches block_tq_v_4b in the header.
 *
 * The block size bump from 32 → 128 amortises the fp16 scale across
 * 4× more elements (2 bytes per 128 vs 8 bytes per 128) and aligns
 * with our K-side TQ_KV_1B blocks so paired cache storage (Step 4.75)
 * can stride a single pair of blocks per K position.
 * ============================================================ */

void quantize_row_tq_v_4b_ref(const float * x, block_tq_v_4b * y, int64_t k) {
    if (!x || !y || k <= 0) return;
    const int qk = TQ_V_4B_BLOCK_SIZE;
    const int nb = (int)(k / qk);

    for (int i = 0; i < nb; i++) {
        /* Per-block max magnitude, matching ggml q4_0's scale derivation. */
        float amax = 0.0f;
        float max  = 0.0f;
        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            const float a = v < 0.0f ? -v : v;
            if (a > amax) { amax = a; max = v; }
        }
        const float d  = max / -8.0f;        /* negative, q4_0 convention     */
        const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

        y[i].d = tq_fp32_to_fp16(d);

        for (int j = 0; j < qk/2; j++) {
            const float x0 = x[i*qk + 0        + j] * id;
            const float x1 = x[i*qk + qk/2     + j] * id;

            /* Round to nearest even, add zero-point 8, clamp to [0, 15]. */
            int q0 = (int)(x0 + 8.5f);
            int q1 = (int)(x1 + 8.5f);
            if (q0 < 0)  q0 = 0;
            if (q0 > 15) q0 = 15;
            if (q1 < 0)  q1 = 0;
            if (q1 > 15) q1 = 15;

            y[i].qs[j] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

void dequantize_row_tq_v_4b(const block_tq_v_4b * x, float * y, int64_t k) {
    const int qk = TQ_V_4B_BLOCK_SIZE;
    const int nb = (int)(k / qk);

    for (int i = 0; i < nb; i++) {
        const float d = tq_fp16_to_fp32(x[i].d);
        for (int j = 0; j < qk/2; j++) {
            const int lo = (x[i].qs[j] & 0x0F) - 8;
            const int hi = (x[i].qs[j] >>   4) - 8;
            y[i*qk + 0        + j] = (float) lo * d;
            y[i*qk + qk/2     + j] = (float) hi * d;
        }
    }
}

/* Fused V dequant + vec mad for the flash attention hot path.
 *
 * SSE4.1 register-resident body. Each block processes 128 elements
 * in the same interleaved low-nibbles-then-high-nibbles layout that
 * ggml_x86_q4_0_unpack_32 uses but with a 128-element rather than
 * 32-element block. The scale `d` and softmax weight `vs` are folded
 * into `scaled = d * vs` so the inner loop is one `_mm_mul_ps` plus
 * one `_mm_add_ps` per 4-lane store of VKQ32.
 *
 * Because ggml-turbo-quant.c is compiled with per-source -march=native
 * (see CMakeLists.txt set_source_files_properties), we can use SSE4.1
 * intrinsics directly here. */
#if defined(__SSE4_1__)

void tq_v_4b_vec_mad_f32(int dv, float * vkq, const block_tq_v_4b * v, float vs) {
    const int qk = TQ_V_4B_BLOCK_SIZE;
    if ((dv % qk) != 0) return; /* caller bug */
    const int nb = dv / qk;

    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i bias8   = _mm_set1_epi32(8);

    for (int b = 0; b < nb; b++) {
        const float scale = tq_fp16_to_fp32(v[b].d) * vs;
        const __m128 vs_vec = _mm_set1_ps(scale);
        const uint8_t * qs = v[b].qs;

        float * out = vkq + b * qk;

        /* Low nibbles: qs[0..63] & 0x0F → 64 values (half of the block). */
        for (int j = 0; j < qk/2; j += 16) {
            /* Load 16 bytes (16 nibble pairs / 16 low nibbles here). */
            const __m128i packed = _mm_loadu_si128((const __m128i *)(qs + j));
            const __m128i lo = _mm_and_si128(packed, mask_0f);

            /* Expand 4 bytes at a time to int32x4, subtract zero-point 8,
             * convert to float, multiply by scale, fmadd into VKQ. */
            __m128i l0 = _mm_sub_epi32(_mm_cvtepu8_epi32(lo), bias8);
            __m128i l1 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(lo, 4)), bias8);
            __m128i l2 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(lo, 8)), bias8);
            __m128i l3 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(lo, 12)), bias8);

            __m128 f0 = _mm_mul_ps(_mm_cvtepi32_ps(l0), vs_vec);
            __m128 f1 = _mm_mul_ps(_mm_cvtepi32_ps(l1), vs_vec);
            __m128 f2 = _mm_mul_ps(_mm_cvtepi32_ps(l2), vs_vec);
            __m128 f3 = _mm_mul_ps(_mm_cvtepi32_ps(l3), vs_vec);

            __m128 a0 = _mm_loadu_ps(out + 0  + j);
            __m128 a1 = _mm_loadu_ps(out + 4  + j);
            __m128 a2 = _mm_loadu_ps(out + 8  + j);
            __m128 a3 = _mm_loadu_ps(out + 12 + j);

            _mm_storeu_ps(out + 0  + j, _mm_add_ps(a0, f0));
            _mm_storeu_ps(out + 4  + j, _mm_add_ps(a1, f1));
            _mm_storeu_ps(out + 8  + j, _mm_add_ps(a2, f2));
            _mm_storeu_ps(out + 12 + j, _mm_add_ps(a3, f3));
        }

        /* High nibbles: (qs[0..63] >> 4) & 0x0F → 64 values (second half). */
        for (int j = 0; j < qk/2; j += 16) {
            const __m128i packed = _mm_loadu_si128((const __m128i *)(qs + j));
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_0f);

            __m128i h0 = _mm_sub_epi32(_mm_cvtepu8_epi32(hi), bias8);
            __m128i h1 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(hi, 4)), bias8);
            __m128i h2 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(hi, 8)), bias8);
            __m128i h3 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(hi, 12)), bias8);

            __m128 f0 = _mm_mul_ps(_mm_cvtepi32_ps(h0), vs_vec);
            __m128 f1 = _mm_mul_ps(_mm_cvtepi32_ps(h1), vs_vec);
            __m128 f2 = _mm_mul_ps(_mm_cvtepi32_ps(h2), vs_vec);
            __m128 f3 = _mm_mul_ps(_mm_cvtepi32_ps(h3), vs_vec);

            __m128 a0 = _mm_loadu_ps(out + qk/2 + 0  + j);
            __m128 a1 = _mm_loadu_ps(out + qk/2 + 4  + j);
            __m128 a2 = _mm_loadu_ps(out + qk/2 + 8  + j);
            __m128 a3 = _mm_loadu_ps(out + qk/2 + 12 + j);

            _mm_storeu_ps(out + qk/2 + 0  + j, _mm_add_ps(a0, f0));
            _mm_storeu_ps(out + qk/2 + 4  + j, _mm_add_ps(a1, f1));
            _mm_storeu_ps(out + qk/2 + 8  + j, _mm_add_ps(a2, f2));
            _mm_storeu_ps(out + qk/2 + 12 + j, _mm_add_ps(a3, f3));
        }
    }
}

#else /* !__SSE4_1__ */

void tq_v_4b_vec_mad_f32(int dv, float * vkq, const block_tq_v_4b * v, float vs) {
    const int qk = TQ_V_4B_BLOCK_SIZE;
    if ((dv % qk) != 0) return;
    const int nb = dv / qk;
    for (int b = 0; b < nb; b++) {
        const float scale = tq_fp16_to_fp32(v[b].d) * vs;
        const uint8_t * qs = v[b].qs;
        float * out = vkq + b * qk;
        for (int j = 0; j < qk/2; j++) {
            const int lo = (qs[j] & 0x0F) - 8;
            const int hi = (qs[j] >>   4) - 8;
            out[j]        += (float) lo * scale;
            out[qk/2 + j] += (float) hi * scale;
        }
    }
}

#endif /* __SSE4_1__ */

/* ============================================================
 * Step 4.75: TQ_KV_FUSED — fused K Hamming + online softmax + V mad.
 *
 * Bundles the work that was previously split between
 * tq_kv_1b_attention_multi (K side) and the per-position softmax+V
 * loop in ggml_compute_forward_flash_attn_ext_f16 (caller). The
 * score scratch is eliminated; K[s] and V[s] are accessed back to
 * back so the L1/L2 prefetcher streams them as a single unit.
 *
 * Numerical equivalence: the online softmax math, the Hamming score
 * formula, and the V mad accumulator update all match the legacy
 * paths bit-for-bit (modulo fp32 reassociation in the SSE4.1 V mad,
 * which already exists in the unfused tq_v_4b_vec_mad_f32).
 * ============================================================ */

/* Inline VKQ32 *= ms scale. SSE4.1 register loop, scalar fallback.
 * dv must be a multiple of 4 (always true for FA: DV=128 or 256). */
static inline void tq_vec_scale_f32_inplace(int dv, float * vkq, float ms) {
#if defined(__SSE4_1__)
    const __m128 ms_vec = _mm_set1_ps(ms);
    int j = 0;
    for (; j + 16 <= dv; j += 16) {
        __m128 a0 = _mm_loadu_ps(vkq + j +  0);
        __m128 a1 = _mm_loadu_ps(vkq + j +  4);
        __m128 a2 = _mm_loadu_ps(vkq + j +  8);
        __m128 a3 = _mm_loadu_ps(vkq + j + 12);
        _mm_storeu_ps(vkq + j +  0, _mm_mul_ps(a0, ms_vec));
        _mm_storeu_ps(vkq + j +  4, _mm_mul_ps(a1, ms_vec));
        _mm_storeu_ps(vkq + j +  8, _mm_mul_ps(a2, ms_vec));
        _mm_storeu_ps(vkq + j + 12, _mm_mul_ps(a3, ms_vec));
    }
    for (; j < dv; j += 4) {
        __m128 a = _mm_loadu_ps(vkq + j);
        _mm_storeu_ps(vkq + j, _mm_mul_ps(a, ms_vec));
    }
#else
    for (int j = 0; j < dv; j++) vkq[j] *= ms;
#endif
}

void tq_kv_fused_attention(
    const float          * query,
    const char           * k_base,
    const char           * v_base,
    size_t                 k_row_stride,
    size_t                 v_row_stride,
    const uint16_t       * mp,
    int                    valid_run,
    int                    DK,
    int                    DV,
    float                  scale,
    float                  slope,
    float                  logit_softcap,
    float                * VKQ32,
    float                * M_inout,
    float                * S_inout)
{
    if (valid_run <= 0) return;

    const int n_blocks_k = DK / TQ_KV_1B_BLOCK_SIZE;
    const int n_blocks_v = DV / TQ_V_4B_BLOCK_SIZE;
    if (n_blocks_k * TQ_KV_1B_BLOCK_SIZE != DK ||
        n_blocks_v * TQ_V_4B_BLOCK_SIZE  != DV ||
        n_blocks_k > TQ_MAX_BLOCKS_PER_KEY) {
        return; /* caller bug; legacy path handles odd dims */
    }
    (void)n_blocks_v;  /* used only for the V-side size check above */

    /* Per-K-block Q precomputation (hoisted out of the position loop).
     * Identical to tq_kv_1b_attention_multi lines 393-410. */
    uint8_t   q_signs[TQ_MAX_BLOCKS_PER_KEY][TQ_KV_1B_BLOCK_SIZE / 8];
    float     q_norm [TQ_MAX_BLOCKS_PER_KEY];
    const float per_block_scale = sqrtf(TQ_PI_2) / (float) TQ_KV_1B_BLOCK_SIZE;

    for (int b = 0; b < n_blocks_k; b++) {
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

    float M = *M_inout;
    float S = *S_inout;

    /* Fused per-position loop. */
    for (int s = 0; s < valid_run; s++) {
        /* Mask gate first — cheapest reject. */
        float mv = 0.0f;
        if (mp) {
            mv = slope * tq_fp16_to_fp32(mp[s]);
            if (mv == -INFINITY) continue;
        }

        /* K-side: Hamming score for position s, sum across DK/128 blocks.
         * Stride from the caller — k_row_stride bytes between consecutive
         * K positions, which may be larger than n_blocks_k * sizeof(block)
         * when multiple KV heads are packed per row (GQA layout). */
        const block_tq_kv_1b * k_row = (const block_tq_kv_1b *)(k_base + (size_t)s * k_row_stride);
        float score = 0.0f;
        for (int b = 0; b < n_blocks_k; b++) {
            int hamming = tq_xor_popcount_16(q_signs[b], k_row[b].signs);
            int agree   = TQ_KV_1B_BLOCK_SIZE - hamming;
            float k_norm = tq_fp16_to_fp32(k_row[b].norm);
            score += q_norm[b] * k_norm * per_block_scale *
                     (float) (2 * agree - TQ_KV_1B_BLOCK_SIZE);
        }

        score = score * scale;
        if (logit_softcap != 0.0f) score = logit_softcap * tanhf(score);
        score += mv;

        /* Online softmax update (matches ops.cpp:8344-8377 fp32 path). */
        float ms = 1.0f;
        float vs = 1.0f;
        if (score > M) {
            const float Mold = M;
            M = score;
            ms = expf(Mold - M);
            tq_vec_scale_f32_inplace(DV, VKQ32, ms);
        } else {
            vs = expf(score - M);
        }

        /* V-side: fused dequant + mad in place. The per-K-position stride
         * comes from the caller (ggml row stride nb[1] of the V tensor).
         * For the GQA layouts the model uses today this is greater than
         * n_blocks_v * sizeof(block_tq_v_4b) because consecutive K positions
         * are interleaved with the other KV head's V data. */
        const block_tq_v_4b * v_row = (const block_tq_v_4b *) (v_base + (size_t) s * v_row_stride);
        tq_v_4b_vec_mad_f32(DV, VKQ32, v_row, vs);

        S = S * ms + vs;
    }

    *M_inout = M;
    *S_inout = S;
}
