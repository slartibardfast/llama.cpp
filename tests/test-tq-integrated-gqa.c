/*
 * test-tq-integrated-gqa.c — Integrated GQA layout diagnostic (all suspects)
 *
 * Replicates the exact Qwen3.5-35B-A3B K cache layout:
 *   head_dim=256, n_kv_heads=2, n_positions=16
 *   n_embd_k_gqa = 512, blocks_per_row = 4, row_bytes = 96
 *
 * Head 0 data from sin(), head 1 from cos() — uncorrelated, so
 * cross-head reads produce obviously wrong scores.
 *
 * Expected result: FAIL at path A vs B (stride bug).
 *                  PASS at path B vs C (quant quality).
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ggml-turbo-quant.h"

#define HEAD_DIM       256
#define N_KV_HEADS     2
#define N_POS          16
#define N_EMBD_K_GQA   (N_KV_HEADS * HEAD_DIM)   /* 512 */
#define BLOCKS_PER_HEAD (HEAD_DIM / TQ_KV_1B_BLOCK_SIZE)  /* 2 */
#define BLOCKS_PER_ROW  (N_KV_HEADS * BLOCKS_PER_HEAD)    /* 4 */

/* Deterministic PRNG producing well-distributed values in [-1, 1]. */
static uint32_t xor_state;
static float xrand(void) {
    xor_state ^= xor_state << 13;
    xor_state ^= xor_state >> 17;
    xor_state ^= xor_state << 5;
    return (float)(xor_state & 0xFFFF) / 32768.0f - 1.0f;
}

int main(void) {
    /* --- Generate raw key data with distinct per-position random seeds ---
     * Each position × head gets its own seed so the data is uncorrelated.
     * This avoids the degenerate sin/cos patterns that produce all-zero
     * TQ scores due to identical sign-hash collisions. */
    float raw_keys[N_POS][N_EMBD_K_GQA];
    for (int pos = 0; pos < N_POS; pos++) {
        /* Head 0: random, norm ~sqrt(256) ≈ 16 */
        xor_state = 0xA0000000u + (uint32_t)pos * 1337;
        for (int i = 0; i < HEAD_DIM; i++) {
            raw_keys[pos][i] = xrand();
        }
        /* Head 1: different seed, scaled ×3 so head leakage is detectable */
        xor_state = 0xB0000000u + (uint32_t)pos * 7919;
        for (int i = 0; i < HEAD_DIM; i++) {
            raw_keys[pos][HEAD_DIM + i] = xrand() * 3.0f;
        }
    }

    /* --- Quantize into GQA-packed KV cache --- */
    block_tq_kv_1b kv_cache[N_POS * BLOCKS_PER_ROW];
    memset(kv_cache, 0, sizeof(kv_cache));

    for (int pos = 0; pos < N_POS; pos++) {
        /* Quantize the full 512-element row (both heads at once). */
        quantize_row_tq_kv_1b_ref(raw_keys[pos], &kv_cache[pos * BLOCKS_PER_ROW], N_EMBD_K_GQA);
    }

    /* --- Generate query for head 0 --- */
    float query[HEAD_DIM];
    xor_state = 0xCAFE0000u;
    for (int i = 0; i < HEAD_DIM; i++) {
        query[i] = xrand();
    }

    /* --- Path A (library batched — simulates what ops.cpp does) ---
     * Passes the base of head 0 and asks for N_POS scores.
     * Internal stride: n_blocks=2 → 48 bytes per position.
     * Correct stride: BLOCKS_PER_ROW=4 → 96 bytes per position. */
    float scores_lib[N_POS];
    tq_kv_1b_attention_multi(query, &kv_cache[0], scores_lib, N_POS, HEAD_DIM,
                             BLOCKS_PER_ROW);

    /* --- Path B (correct reference — per-position, manual stride) --- */
    float scores_ref[N_POS];
    for (int pos = 0; pos < N_POS; pos++) {
        const block_tq_kv_1b * head0 = &kv_cache[pos * BLOCKS_PER_ROW];
        tq_kv_1b_attention_multi(query, head0, &scores_ref[pos], 1, HEAD_DIM, 0);
    }

    /* --- Path C (FP32 dot product reference) --- */
    float fp_dots[N_POS];
    for (int pos = 0; pos < N_POS; pos++) {
        float dot = 0.0f;
        for (int i = 0; i < HEAD_DIM; i++) {
            dot += query[i] * raw_keys[pos][i];  /* head 0 only */
        }
        fp_dots[pos] = dot;
    }

    /* --- Check A vs B (stride correctness) --- */
    int n_stride_fail = 0;
    printf("=== Stride check: batched (A) vs per-position (B) ===\n");
    printf("%-5s  %12s  %12s  %8s\n", "pos", "batched(A)", "ref(B)", "match");
    for (int pos = 0; pos < N_POS; pos++) {
        float diff = fabsf(scores_lib[pos] - scores_ref[pos]);
        int ok = (diff < 1e-4f);
        printf("  %3d  %12.6f  %12.6f  %s\n",
               pos, scores_lib[pos], scores_ref[pos],
               ok ? "  OK" : "FAIL");
        if (!ok) n_stride_fail++;
    }

    /* --- Summary --- */
    printf("\nStride check (A vs B): %d/%d positions match. %s\n",
           N_POS - n_stride_fail, N_POS,
           n_stride_fail ? "FAIL — K stride bug detected." : "PASS");

    if (n_stride_fail > 0) {
        return 1;
    }
    printf("PASS: GQA stride is correct.\n");
    return 0;

    (void)fp_dots;  /* reserved for future quality checks */
}
