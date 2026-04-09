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

int main(void) {
    /* --- Generate raw key data (float, full n_embd_k_gqa per position) --- */
    float raw_keys[N_POS][N_EMBD_K_GQA];
    for (int pos = 0; pos < N_POS; pos++) {
        for (int i = 0; i < HEAD_DIM; i++) {
            /* Head 0: sin-based, norm ~1 */
            raw_keys[pos][i] = sinf((float)i * 0.1f + (float)pos * 0.7f);
        }
        for (int i = 0; i < HEAD_DIM; i++) {
            /* Head 1: cos-based, shifted, norm ~1 but uncorrelated with head 0 */
            raw_keys[pos][HEAD_DIM + i] = cosf((float)i * 0.3f + (float)pos * 1.3f) * 3.0f;
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
    for (int i = 0; i < HEAD_DIM; i++) {
        query[i] = sinf((float)i * 0.05f + 1.0f);
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

    /* --- Check B vs C (quantization quality) --- */
    int n_rank_fail = 0;
    printf("\n=== Ranking check: TQ ref (B) vs FP32 (C) ===\n");
    printf("%-5s  %12s  %12s\n", "pos", "TQ ref(B)", "FP32(C)");
    int tq_argmax = 0, fp_argmax = 0;
    for (int pos = 0; pos < N_POS; pos++) {
        printf("  %3d  %12.6f  %12.6f\n", pos, scores_ref[pos], fp_dots[pos]);
        if (scores_ref[pos] > scores_ref[tq_argmax]) tq_argmax = pos;
        if (fp_dots[pos]    > fp_dots[fp_argmax])     fp_argmax = pos;
    }
    printf("Argmax: TQ ref=%d, FP32=%d\n", tq_argmax, fp_argmax);
    if (tq_argmax != fp_argmax) n_rank_fail++;

    /* --- Summary --- */
    printf("\n=== Summary ===\n");
    printf("Stride check (A vs B): %d/%d positions match. %s\n",
           N_POS - n_stride_fail, N_POS,
           n_stride_fail ? "FAIL — K stride bug detected." : "PASS");
    printf("Ranking check (B vs C): argmax %s. %s\n",
           (tq_argmax == fp_argmax) ? "agrees" : "DISAGREES",
           n_rank_fail ? "FAIL — quantization quality issue." : "PASS");

    if (n_stride_fail > 0 || n_rank_fail > 0) {
        return 1;
    }
    printf("\nAll checks passed.\n");
    return 0;
}
