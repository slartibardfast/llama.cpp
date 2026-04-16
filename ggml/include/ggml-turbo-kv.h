/*
 * ggml-turbo-kv.h — port of quantumaikr/quant.cpp v0.8.0 turbo_kv_4b
 *
 * Real TurboQuant (ICLR 2026 / HIGGS Nov 2024 algorithm family):
 *   Random Hadamard Transform + 16-entry Lloyd-Max-Gaussian codebook.
 *
 * Block layout (72 bytes per 128 elements, 4.5 bpe):
 *
 *   struct block_turbo_kv_4b {
 *       uint16_t norm;            // fp16, ||x||_2 before normalization
 *       uint16_t residual_norm;   // unused (reserved for future composite residual)
 *       uint16_t inv_std_fp16;    // fp16, = 2.7326 / max(|rotated|), per block
 *       uint16_t _pad;            // alignment
 *       uint8_t  mse_indices[64]; // 4-bit packed, 2 indices/byte, LSB-first
 *   };
 *
 * Pipeline (per 128-element block):
 *
 *   quantize:   norm <- ||x||
 *               r    <- x / norm
 *               r    <- RHT(r, seed=TURBO_KV_DEFAULT_SEED)
 *               inv_std <- 2.7326 / max(|r|)
 *               mse_indices[i] <- argmin_c |codebook[c] - r[i] * inv_std|
 *
 *   dequant:    r[i] <- codebook[mse_indices[i]] / inv_std
 *               r    <- RHT^{-1}(r)
 *               x[i] <- r[i] * norm
 *
 *   attention:  pre-rotate query once:   q_rot = RHT(query)
 *               per block: compute <q_rot, dequant_rotated_k_block> * norm
 *               -- inverse RHT not needed on K side because <a,b> = <RHT(a),RHT(b)>
 *                  for orthogonal RHT
 *
 * The 16-entry Lloyd-Max-Gaussian codebook is a static constant — no calibration
 * data needed. See tq_codebook.c in quant.cpp for provenance (Max 1960 tables).
 *
 * Source: github.com/quantumaikr/quant.cpp v0.8.0, Apache-2.0 licensed.
 * Vendored with attribution into the polaris-hybrid-cpu-opt branch to support
 * CPU-bound KV compression on Qwen3.5-35B-A3B and similar hybrid DeltaNet/MoE
 * models where KV bandwidth matters at long context.
 */

#ifndef GGML_TURBO_KV_H
#define GGML_TURBO_KV_H

#include "ggml.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TURBO_KV_BLOCK_SIZE 128    /* elements per block */
#define TURBO_KV_4B_BYTES   72     /* bytes per block */

/* Random Hadamard Transform seed (quant.cpp TKV_DEFAULT_SEED).
 * Shared between quantize, dequantize, and query pre-rotation so that
 * <query, key>  ==  <RHT(query), RHT(key)>.
 */
#define TURBO_KV_DEFAULT_SEED 0x12345678u

/* Largest Lloyd-Max centroid for N(0,1) at 4-bit precision.
 * Used to scale rotated values so max_abs maps to ±CENT_4BIT_MAX. */
#define TURBO_KV_4B_CENT_MAX 2.7326f

typedef struct {
    uint16_t norm;            /* fp16, ||x||_2 of the original vector */
    uint16_t residual_norm;   /* unused (kept for layout with future residual) */
    uint16_t inv_std_fp16;    /* fp16, = CENT_MAX / max(|rotated|), per block */
    uint16_t _pad;            /* alignment */
    uint8_t  mse_indices[TURBO_KV_BLOCK_SIZE / 2]; /* 4-bit packed, 2 per byte */
} block_turbo_kv_4b;

/* Compile-time size check: 2+2+2+2+64 = 72 */
typedef char turbo_kv_4b_check_block_size
    [(sizeof(block_turbo_kv_4b) == TURBO_KV_4B_BYTES) ? 1 : -1];

/* ============================================================
 * FP16 / FP32 conversion helpers (used by all turbo_* types)
 * ============================================================ */
float    turbo_kv_fp16_to_fp32(uint16_t h);
uint16_t turbo_kv_fp32_to_fp16(float f);

/* ============================================================
 * Lloyd-Max codebooks (16 entries, symmetric, 4-bit)
 *
 * turbo_kv_4b_codebook: Lloyd-Max optimal for N(0,1) (Max 1960 tables).
 *   Used by TURBO_KV_4B (KV cache), where the post-RHT distribution varies
 *   by sequence and token — Gaussian is a safe static choice.
 *
 * turbo_4b_weight_codebook: Lloyd-Max optimal for empirical post-RHT weight
 *   distribution (Qwen3.5-0.8B, 772M values). Kurtosis=2.96 (sub-Gaussian),
 *   tails ~0.2 lighter than Gaussian. 5.26% MSE improvement over Gaussian
 *   codebook on real weight data at d=128.
 * ============================================================ */
extern const float turbo_kv_4b_codebook[16];
extern const float turbo_4b_weight_codebook[16];

/* ============================================================
 * Random Hadamard Transform (O(n log n) butterfly, in-place)
 *
 * n must be a power of 2. The transform is:
 *
 *   data[i] <- data[i] * random_sign(seed, i)
 *   data    <- walsh_hadamard(data)
 *   data[i] <- data[i] / sqrt(n)
 *
 * Self-inverse except for the sign step: to invert, apply the scale then
 * walsh_hadamard then the same sign flip.
 * ============================================================ */
void turbo_kv_rht_forward(float * data, int n, uint32_t seed);
void turbo_kv_rht_inverse(float * data, int n, uint32_t seed);

/* ============================================================
 * Quantize / dequantize API
 *
 * k must be a multiple of TURBO_KV_BLOCK_SIZE (128).
 * ============================================================ */
void quantize_row_turbo_kv_4b_ref(const float * x, block_turbo_kv_4b * y, int64_t k);
void dequantize_row_turbo_kv_4b  (const block_turbo_kv_4b * x, float * y, int64_t k);

/* ============================================================
 * Vector dot product: <f32 query, quantized key>
 *
 * Called by ggml mul_mat and flash-attention kq_vec_dot paths. This is
 * the hot inner loop during attention. Takes a pre-rotated query (see
 * turbo_kv_rotate_query_once below) and a pointer to one K block.
 *
 * n is the head dimension in elements (must be <= TURBO_KV_BLOCK_SIZE).
 * ============================================================ */
void ggml_vec_dot_turbo_kv_4b_f32(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by,
    int nrc);

/* ============================================================
 * Query pre-rotation helper.
 *
 * Called once per Q row per attention layer. The result is a rotated
 * f32 buffer that can be dotted directly against K blocks without any
 * per-block inverse rotation (RHT is orthogonal).
 *
 * dim must be a power of 2 <= TURBO_KV_BLOCK_SIZE; if dim < BK, the
 * upper entries of out are zero-padded.
 * ============================================================ */
void turbo_kv_rotate_query(const float * q, float * out, int dim);

/* ============================================================
 * Batched attention: compute scores for all K positions in one call.
 *
 * Rotates Q ONCE, then loops over valid_count K positions using the
 * SSSE3 inner kernel. Eliminates the per-call rotation overhead that
 * caused +52% wall-clock regression on tool-call workloads.
 *
 * Mirrors tq_kv_1b_attention_multi in ggml-turbo-quant.h. The FA path
 * in ops.cpp calls this once per thread with a slice of K. The non-FA
 * vec_dot wrapper calls it with valid_count=1 as a fallback.
 *
 * @param query         Raw f32 query vector (head_dim elements, NOT pre-rotated)
 * @param kv_cache      Pointer to the first K block for position 0 in this head
 * @param scores        Output: one f32 score per K position (valid_count elements)
 * @param valid_count   Number of K positions to process
 * @param head_dim      Head dimension (e.g. 256 for Qwen3.5), multiple of 128
 * @param k_stride_blocks  Number of turbo_kv_4b blocks between consecutive K
 *                         positions in the cache. For dense layout: head_dim/128.
 *                         For GQA interleave: n_embd_k_gqa / 128. Pass 0 for
 *                         the default (= head_dim / 128).
 * ============================================================ */
void turbo_kv_4b_attention_multi(
    const float              * query,
    const block_turbo_kv_4b  * kv_cache,
    float                    * scores,
    int                        valid_count,
    int                        head_dim,
    int                        k_stride_blocks);

/* ============================================================
 * TURBO_4B — RHT + Lloyd-Max codebook for WEIGHT tensors
 *
 * Same algorithm as TURBO_KV_4B (RHT + 16-entry codebook) but
 * optimized for offline weight quantization:
 *   - Drops residual_norm + _pad fields (68 bytes vs 72)
 *   - Two block sizes: 128 (primary) and 64 (fallback for ne[0] % 128 != 0)
 *   - vec_dot does full inverse RHT per block (no query pre-rotation trick)
 *   - Fused variant: pre-rotate activation, dot in RHT space
 * ============================================================ */

#define TURBO_4B_BLOCK_SIZE    128   /* elements per block (primary) */
#define TURBO_4B_BYTES         68    /* 2+2+64 bytes per 128-elem block */
#define TURBO_4B_S_BLOCK_SIZE  64    /* elements per block (small fallback) */
#define TURBO_4B_S_BYTES       36    /* 2+2+32 bytes per 64-elem block */

typedef struct {
    uint16_t norm;                              /* fp16, ||x||_2 of original block */
    uint16_t inv_std;                           /* fp16, = CENT_MAX / max(|rotated|) */
    uint8_t  qs[TURBO_4B_BLOCK_SIZE / 2];       /* 4-bit packed codebook indices */
} block_turbo_4b;

typedef char turbo_4b_check_size
    [(sizeof(block_turbo_4b) == TURBO_4B_BYTES) ? 1 : -1];

typedef struct {
    uint16_t norm;                              /* fp16, ||x||_2 of original block */
    uint16_t inv_std;                           /* fp16, = CENT_MAX / max(|rotated|) */
    uint8_t  qs[TURBO_4B_S_BLOCK_SIZE / 2];     /* 4-bit packed codebook indices */
} block_turbo_4b_s;

typedef char turbo_4b_s_check_size
    [(sizeof(block_turbo_4b_s) == TURBO_4B_S_BYTES) ? 1 : -1];

/* Quantize / dequantize API for weight tensors */
void quantize_row_turbo_4b_ref  (const float * GGML_RESTRICT x, block_turbo_4b   * GGML_RESTRICT y, int64_t k);
void dequantize_row_turbo_4b    (const block_turbo_4b   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void quantize_row_turbo_4b_s_ref(const float * GGML_RESTRICT x, block_turbo_4b_s * GGML_RESTRICT y, int64_t k);
void dequantize_row_turbo_4b_s  (const block_turbo_4b_s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

/* imatrix-aware quantization wrappers (ggml_quantize_chunk API) */
size_t quantize_turbo_4b  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                            int64_t nrows, int64_t n_per_row, const float * quant_weights);
size_t quantize_turbo_4b_s(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                            int64_t nrows, int64_t n_per_row, const float * quant_weights);

/* vec_dot: <quantized weight row, f32 activation> */
void ggml_vec_dot_turbo_4b_f32(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by,
    int nrc);

void ggml_vec_dot_turbo_4b_s_f32(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by,
    int nrc);

#ifdef __cplusplus
}
#endif

#endif /* GGML_TURBO_KV_H */
