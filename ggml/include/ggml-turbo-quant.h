/**
 * ggml-turbo-quant.h -- quant.cpp 1-bit KV cache quantization for llama.cpp
 *
 * Apache 2.0 License, QuantumAI Inc.
 *
 * Self-contained implementation of quant.cpp 1-bit KV cache compression.
 * Algorithm: L2-normalize -> Random Hadamard Transform -> sign extraction.
 * Attention: XOR + popcount Hamming distance -> inner product estimator.
 *
 * Reference: quant.cpp (arXiv 2504.19874)
 *   - 1-bit per dimension with RHT decorrelation
 *   - Theoretical attention cosine similarity: 2/pi ~ 0.637
 *   - Compression: 20 bytes per 128 elements (1.25 bpw including metadata)
 *
 * Usage in llama.cpp:
 *   --cache-type-k tq_kv_1b   (for key cache)
 *   --cache-type-v tq_kv_1b   (for value cache, though key-only is recommended)
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Block definition: quant.cpp 1-bit KV cache
 *
 * 20 bytes per 128 elements = 1.25 bits per element (with metadata)
 * Pure sign bits = 1.0 bpw, metadata overhead = 0.25 bpw
 *
 * Layout:
 *   norm     (2B) - L2 norm of original vector, stored as FP16
 *   _pad     (2B) - alignment padding (reserved for future use)
 *   rht_seed (4B) - RHT random seed for inverse transform
 *   signs   (16B) - 128 sign bits, LSB-first packing
 *
 * Total: 24 bytes per 128 elements
 * Compression vs FP16: 256 bytes / 24 bytes = 10.7x
 * Compression vs FP32: 512 bytes / 24 bytes = 21.3x
 * ============================================================ */

#define TQ_KV_1B_BLOCK_SIZE 128

typedef struct {
    uint16_t norm;                              /* L2 norm in FP16               */
    uint16_t _pad;                              /* alignment padding             */
    uint32_t rht_seed;                          /* RHT seed for inverse          */
    uint8_t  signs[TQ_KV_1B_BLOCK_SIZE / 8];   /* 128 sign bits = 16 bytes      */
} block_tq_kv_1b;

/* Compile-time size check: 2 + 2 + 4 + 16 = 24 bytes */
typedef char tq_check_block_size[(sizeof(block_tq_kv_1b) == 24) ? 1 : -1];

/* ============================================================
 * Block definition: TurboQuant V 4-bit (symmetric, 128 elements)
 *
 * 66 bytes per 128 elements = 4.125 bits per element (with metadata).
 *
 * Layout:
 *   d       (2B) - per-block scale (FP16, negative, following q4_0 semantics)
 *   qs     (64B) - 128 signed 4-bit values, packed as nibble pairs
 *                   low nibble at qs[j] & 0x0F, high nibble at qs[j] >> 4
 *
 * Quantization:
 *   d = -max(abs(x_min), abs(x_max)) / 8          (matches q4_0 sign convention)
 *   q = clamp(round(x / d + 8), 0, 15)            (zero-point 8, unsigned storage)
 *
 * Dequantization:
 *   x ≈ (q - 8) * d                               (symmetric, centered at zero)
 *
 * Layout matches ggml q4_0 semantics but with 128-element blocks (vs
 * q4_0's 32-element) to amortise the scale over 4x more elements and
 * match our K-side TQ_KV_1B block size for paired cache storage.
 * ============================================================ */

#define TQ_V_4B_BLOCK_SIZE 128

typedef struct {
    uint16_t d;                             /* per-block scale in FP16 */
    uint8_t  qs[TQ_V_4B_BLOCK_SIZE / 2];    /* 128 nibbles = 64 bytes */
} block_tq_v_4b;

/* Compile-time size check: 2 + 64 = 66 bytes */
typedef char tq_v_4b_check_block_size[(sizeof(block_tq_v_4b) == 66) ? 1 : -1];

/* ============================================================
 * Public API (matches llama.cpp quantize/dequantize convention)
 *
 * k: number of elements (must be multiple of TQ_KV_1B_BLOCK_SIZE)
 * ============================================================ */

/**
 * Quantize a row of float values to 1-bit quant.cpp KV blocks.
 *
 * Pipeline: L2-normalize -> RHT (Walsh-Hadamard + random signs) -> sign extraction.
 *
 * @param x   Input float array (k elements)
 * @param y   Output block array (k / TQ_KV_1B_BLOCK_SIZE blocks)
 * @param k   Number of elements (must be multiple of 128)
 */
void quantize_row_tq_kv_1b_ref(const float * x, block_tq_kv_1b * y, int64_t k);

/**
 * Dequantize 1-bit quant.cpp KV blocks back to float.
 *
 * Pipeline: sign -> scale by sqrt(2/pi)/sqrt(dim) -> inverse RHT -> scale by norm.
 * Note: This is a rough reconstruction. The real value of 1-bit is in Hamming attention.
 *
 * @param x   Input block array
 * @param y   Output float array (k elements)
 * @param k   Number of elements (must be multiple of 128)
 */
void dequantize_row_tq_kv_1b(const block_tq_kv_1b * x, float * y, int64_t k);

/**
 * Compute attention scores between a query and quantized KV cache.
 *
 * Uses XOR + popcount Hamming distance for ultra-fast attention:
 *   score = q_norm * k_norm * sqrt(pi/2) / dim * (2*agree - dim)
 *
 * @param query     Float query vector (head_dim elements)
 * @param kv_cache  Array of quantized key blocks (seq_len blocks)
 * @param scores    Output attention scores (seq_len elements)
 * @param seq_len   Number of keys in the cache
 * @param head_dim  Dimension of each head (must be <= 128)
 */
void tq_kv_1b_attention(const float * query, const block_tq_kv_1b * kv_cache,
                         float * scores, int seq_len, int head_dim);

/**
 * Multi-block Hamming attention for head_dim > 128.
 *
 * Each key in the cache is stored as ceil(head_dim/128) consecutive blocks.
 * The query is split into blocks too. Score for each (query, key) pair is
 * the sum of per-block contributions, mathematically equivalent to
 * <q, k> = sum_b <q_b, k_b>.
 *
 * On x86-64 with SSE4.2, this uses _mm_xor_si128 + POPCNT for ~6 cycles
 * per K position per block (vs ~30 cycles scalar). Westmere has POPCNT.
 *
 * The valid_count parameter must be the number of K positions that are
 * NOT masked. Allocated KV slots beyond this are skipped to avoid wasted
 * DRAM bandwidth (a 64K-allocated cache with 4K filled would otherwise
 * read 16x the necessary data and dominate the savings from compression).
 *
 * @param query        Float query vector (head_dim elements)
 * @param kv_cache     Block array (seq_capacity * (head_dim/128) blocks)
 * @param scores       Output scores (valid_count, the rest are not touched)
 * @param valid_count  Number of contiguous valid K positions starting at 0
 * @param head_dim     Head dimension (multiple of 128, max 512)
 */
void tq_kv_1b_attention_multi(const float * query,
                               const block_tq_kv_1b * kv_cache,
                               float * scores, int valid_count, int head_dim);

/* ============================================================
 * TurboQuant V 4-bit API
 * ============================================================ */

/**
 * Quantize a row of float V values into TQ_V_4B blocks.
 * k must be a multiple of TQ_V_4B_BLOCK_SIZE (128).
 *
 * Symmetric q4_0-style scheme:
 *   d = -max(abs(x)) / 8
 *   q = clamp(round(x / d + 8), 0, 15)
 */
void quantize_row_tq_v_4b_ref(const float * x, block_tq_v_4b * y, int64_t k);

/**
 * Dequantize a row of TQ_V_4B blocks back to float. Used by ggml's
 * type_traits to_float slot for generic code paths (our hot path
 * bypasses this by calling the fused mad helper directly).
 */
void dequantize_row_tq_v_4b(const block_tq_v_4b * x, float * y, int64_t k);

/**
 * Fused V dequant + vector mad for the flash attention hot loop.
 *
 * Computes: VKQ32[0..dv-1] += dequant(v[0..dv/128 blocks]) * vs
 *
 * This avoids the dequant-then-mad round trip and the intermediate
 * fp32 V buffer (per-thread scratch). The dequant result is kept
 * in registers and directly fmadd'd into VKQ32. On Westmere this
 * is a register-resident SSE4.1 loop.
 *
 * dv must be a multiple of TQ_V_4B_BLOCK_SIZE (128).
 */
void tq_v_4b_vec_mad_f32(int dv, float * vkq, const block_tq_v_4b * v, float vs);

/* ============================================================
 * Fused K Hamming + online softmax + V dequant-mad for one FA row.
 *
 * Bundles the work that was previously split between
 * tq_kv_1b_attention_multi (K Hamming, batched into a thread-local
 * score scratch) and the per-position softmax+V loop in
 * ggml_compute_forward_flash_attn_ext_f16. Eliminates the score
 * scratch entirely and walks the valid K range in a single pass,
 * reading K[s] and V[s] back to back per position.
 *
 * `k_base` and `v_base` are byte-pointers into the K and V cache
 * tensors, already offset by ic_start and the head/batch indices.
 * `v_row_stride` is the per-K-position byte stride within a single
 * head (== ggml row stride nb[1]). The K side uses the legacy
 * dense-block stride for byte-equivalence with tq_kv_1b_attention_multi.
 *
 * Caller initializes VKQ32[0..DV-1] = 0, *M_inout = -INFINITY,
 * *S_inout = 0 before the call (matching the existing FA loop).
 *
 * Mask: per-position fp16 mask from the FA op, applied as
 *   score += slope * fp16_to_fp32(mp[s])
 * Positions where the masked score is -INFINITY are skipped at
 * zero cost. Pass mp = NULL for no mask.
 *
 * DK and DV must both be multiples of 128.
 * ============================================================ */
void tq_kv_fused_attention(
    const float          * query,
    const char           * k_base,        /* K cache base, already offset      */
    const char           * v_base,        /* V cache base, already offset      */
    size_t                 v_row_stride,  /* bytes per K position in V cache   */
    const uint16_t       * mp,            /* fp16 mask, NULL for no mask       */
    int                    valid_run,
    int                    DK,
    int                    DV,
    float                  scale,
    float                  slope,
    float                  logit_softcap, /* 0.0f for no softcap               */
    float                * VKQ32,
    float                * M_inout,
    float                * S_inout);

#ifdef __cplusplus
}
#endif
