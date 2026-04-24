/*
 * ggml-turbo-kv.h — port of quantumaikr/quant.cpp v0.8.0 turbo_kv_4b
 *
 * Real TurboQuant (ICLR 2026 / HIGGS Nov 2024 algorithm family):
 *   Random Hadamard Transform + 16-entry Lloyd-Max-Gaussian codebook.
 *
 * Block layout (72 bytes per 128 elements, 4.5 bpe):
 *
 *   struct block_turbo_kv_4b {
 *       float    norm;            // fp32, ||x||_2 before normalization
 *       float    inv_std;         // fp32, = 2.7326 / max(|rotated|), per block
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
    float    norm;            /* fp32, ||x||_2 of the original vector */
    float    inv_std;         /* fp32, = CENT_MAX / max(|rotated|), per block */
    uint8_t  mse_indices[TURBO_KV_BLOCK_SIZE / 2]; /* 4-bit packed, 2 per byte */
} block_turbo_kv_4b;

/* Compile-time size check: 4+4+64 = 72.
 * Earlier layout stored norm + inv_std as fp16 plus two 16-bit words
 * (residual_norm reserved for future residual encoding, plus an
 * alignment pad). Repacked to fp32 scales: same 72 bytes total,
 * eliminates the fp16 round-trip (~1e-3 relative precision loss) on
 * both per-block scales. See PHASE25 notes. */
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
 * All TURBO weight types use the same Gaussian codebooks by default.
 * Model-specific codebooks can be computed via llama-turbo-codebook
 * and loaded at quantize time.
 * ============================================================ */
extern const float turbo_kv_4b_codebook[16];

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

/* Prepare one block: run steps 1-4 of the quantize pipeline
 * (L2 norm, normalize, RHT, max_abs → inv_std). Writes block->norm
 * and block->inv_std (both fp32); fills rotated_out[0..TURBO_KV_BLOCK_SIZE-1]
 * with the rotated values (zero-padded where dim < block size). Returns
 * inv_std as fp32 for the caller's Step 5.
 *
 * Exposed so ggml-cpu's AVX2 argmin variant can share this prep
 * with the scalar ggml-base reference. The scalar
 * quantize_row_turbo_kv_4b_ref uses this helper internally.
 */
float turbo_kv_4b_prepare_block(
    const float * src, int dim, block_turbo_kv_4b * block, float * rotated_out);

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

/* ---- TURBO_2B: 2-bit, 4-level Lloyd-Max (tq_codebook.c CODEBOOK_2BIT) ---- */

#define TURBO_2B_BLOCK_SIZE  128
#define TURBO_2B_BYTES       36   /* 2+2+32 bytes per 128-elem block (2 bits/elem) */

typedef struct {
    uint16_t norm;
    uint16_t inv_std;
    uint8_t  qs[TURBO_2B_BLOCK_SIZE / 4]; /* 2-bit packed, 4 per byte */
} block_turbo_2b;

typedef char turbo_2b_check_size
    [(sizeof(block_turbo_2b) == TURBO_2B_BYTES) ? 1 : -1];

/* ---- TURBO_3B: 3-bit, 8-level Lloyd-Max (tq_codebook.c CODEBOOK_3BIT) ---- */

#define TURBO_3B_BLOCK_SIZE  128
#define TURBO_3B_BYTES       52   /* 2+2+48 bytes per 128-elem block (3 bits/elem) */

typedef struct {
    uint16_t norm;
    uint16_t inv_std;
    uint8_t  qs[48];              /* 3-bit packed: 8 elems → 3 bytes (bit-stream) */
} block_turbo_3b;

typedef char turbo_3b_check_size
    [(sizeof(block_turbo_3b) == TURBO_3B_BYTES) ? 1 : -1];

/* ---- TURBO_5B: 5-bit, 32-level Lloyd-Max (tq_codebook.c CODEBOOK_5BIT) ---- */

#define TURBO_5B_BLOCK_SIZE  128
#define TURBO_5B_BYTES       84   /* 2+2+80 bytes per 128-elem block (5 bits/elem) */

typedef struct {
    uint16_t norm;
    uint16_t inv_std;
    uint8_t  qs[80];              /* 5-bit packed: 8 elems → 5 bytes (bit-stream) */
} block_turbo_5b;

typedef char turbo_5b_check_size
    [(sizeof(block_turbo_5b) == TURBO_5B_BYTES) ? 1 : -1];

/* Published Lloyd-Max Gaussian codebooks from tq_codebook.c (Max 1960) */
extern const float turbo_codebook_2bit[4];
extern const float turbo_codebook_3bit[8];
extern const float turbo_codebook_5bit[32];

/* Runtime codebook override. Set a custom Lloyd-Max codebook (e.g.
   imatrix-weighted centroids from the llama-turbo-codebook tool) before
   invoking the quantize/dequant paths, and clear with NULL to restore the
   published Gaussian defaults. Affects both CPU quantize (index assignment)
   and CPU dequant/vec_dot (reconstruction) — GPU dequant uses a separate
   per-device storage buffer populated from the GGUF codebook tensors.

   Centroids MUST be in [-cent_max, cent_max] range, matching the per-bitrate
   published convention (2-bit: 1.5104, 3-bit: 2.1520, 4-bit: 2.7326,
   5-bit: 1.9956). The llama-turbo-codebook tool rescales its Lloyd-Max
   output to this convention before writing. A codebook at a different scale
   will produce garbage because quantize_block_turbo normalizes each block
   to [-cent_max, cent_max] before codebook lookup. */
void turbo_set_quantize_codebook(int bits, const float * centroids);

/* Quantize / dequantize API for weight tensors */
void quantize_row_turbo_2b_ref  (const float * GGML_RESTRICT x, block_turbo_2b   * GGML_RESTRICT y, int64_t k);
void dequantize_row_turbo_2b    (const block_turbo_2b   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void quantize_row_turbo_3b_ref  (const float * GGML_RESTRICT x, block_turbo_3b   * GGML_RESTRICT y, int64_t k);
void dequantize_row_turbo_3b    (const block_turbo_3b   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void quantize_row_turbo_4b_ref  (const float * GGML_RESTRICT x, block_turbo_4b   * GGML_RESTRICT y, int64_t k);
void dequantize_row_turbo_4b    (const block_turbo_4b   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void quantize_row_turbo_4b_s_ref(const float * GGML_RESTRICT x, block_turbo_4b_s * GGML_RESTRICT y, int64_t k);
void dequantize_row_turbo_4b_s  (const block_turbo_4b_s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void quantize_row_turbo_5b_ref  (const float * GGML_RESTRICT x, block_turbo_5b   * GGML_RESTRICT y, int64_t k);
void dequantize_row_turbo_5b    (const block_turbo_5b   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

/* ============================================================
 * E8P lattice vector quantizer (8D, 256-entry D8-hat codebook)
 *
 * At 4 bpw: RVQ with two 16-bit E8P codes per group of 8 elements.
 * 32 bits per group × 16 groups = 64 bytes = same qs[64] storage.
 * ============================================================ */
void     e8p_decode_16bit(uint16_t code, float * out);
uint16_t e8p_encode_16bit(const float * x);
void     e8p_decode_rvq4bit(uint32_t code, float * out);
uint32_t e8p_encode_rvq4bit(const float * x);
uint32_t e8p_encode_rvq4bit_weighted(const float * x, const float * weights);

/* imatrix-aware quantization wrappers (ggml_quantize_chunk API) */
size_t quantize_turbo_2b  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                            int64_t nrows, int64_t n_per_row, const float * quant_weights);
size_t quantize_turbo_3b  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                            int64_t nrows, int64_t n_per_row, const float * quant_weights);
size_t quantize_turbo_4b  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                            int64_t nrows, int64_t n_per_row, const float * quant_weights);
size_t quantize_turbo_4b_s(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                            int64_t nrows, int64_t n_per_row, const float * quant_weights);
size_t quantize_turbo_5b  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                            int64_t nrows, int64_t n_per_row, const float * quant_weights);

/* vec_dot: <quantized weight row, f32 activation> */
void ggml_vec_dot_turbo_2b_f32(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by,
    int nrc);

void ggml_vec_dot_turbo_3b_f32(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by,
    int nrc);

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

void ggml_vec_dot_turbo_5b_f32(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by,
    int nrc);

#ifdef __cplusplus
}
#endif

#endif /* GGML_TURBO_KV_H */
