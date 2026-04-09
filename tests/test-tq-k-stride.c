/*
 * test-tq-k-stride.c — Diagnostic for Suspect #1: K stride bug
 *
 * Constructs a KV cache with 2 KV heads × 8 positions.
 * Head 0 keys have norm ~1.0, head 1 keys have norm ~5.0.
 * If the internal stride in tq_kv_1b_attention_multi is wrong,
 * odd-indexed scores will reflect head 1's norm (~5x the expected
 * magnitude) instead of the correct next-position head 0 score.
 *
 * Expected result: FAIL (the stride bug is present in the current code).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ggml-turbo-quant.h"

#define HEAD_DIM   256
#define N_KV_HEADS 2
#define N_POS      8
#define BLOCKS_PER_HEAD (HEAD_DIM / TQ_KV_1B_BLOCK_SIZE)  /* 2 */
#define BLOCKS_PER_ROW  (N_KV_HEADS * BLOCKS_PER_HEAD)    /* 4 */

/* Simple deterministic PRNG (xorshift32). */
static uint32_t xor_state = 0xDEADBEEF;
static float xrand(void) {
    xor_state ^= xor_state << 13;
    xor_state ^= xor_state >> 17;
    xor_state ^= xor_state << 5;
    return (float)(xor_state & 0xFFFF) / 65536.0f - 0.5f;
}

int main(void) {
    /* --- Allocate KV cache: N_POS rows × BLOCKS_PER_ROW blocks --- */
    block_tq_kv_1b kv_cache[N_POS * BLOCKS_PER_ROW];
    memset(kv_cache, 0, sizeof(kv_cache));

    /* --- Generate and quantize keys --- */
    float key_buf[HEAD_DIM];

    for (int pos = 0; pos < N_POS; pos++) {
        for (int head = 0; head < N_KV_HEADS; head++) {
            /* Head 0: norm ~1.0, Head 1: norm ~5.0 */
            float scale = (head == 0) ? 1.0f : 5.0f;
            xor_state = 0xDEADBEEF + (uint32_t)(pos * 1000 + head * 100);
            for (int i = 0; i < HEAD_DIM; i++) {
                key_buf[i] = xrand() * scale;
            }

            /* Quantize into the correct cache position.
             * Row layout: [h0_blk0, h0_blk1, h1_blk0, h1_blk1]
             * Position s, head h → block index s*BLOCKS_PER_ROW + h*BLOCKS_PER_HEAD */
            block_tq_kv_1b * dst = &kv_cache[pos * BLOCKS_PER_ROW + head * BLOCKS_PER_HEAD];
            quantize_row_tq_kv_1b_ref(key_buf, dst, HEAD_DIM);
        }
    }

    /* --- Generate a query --- */
    float query[HEAD_DIM];
    xor_state = 0xCAFEBABE;
    for (int i = 0; i < HEAD_DIM; i++) query[i] = xrand();

    /* --- Path A: batched call (what the buggy code does) ---
     * Passes the base of head 0 and asks for N_POS scores.
     * The function internally strides by n_blocks=2 (48 bytes) per position,
     * but the real per-position stride is BLOCKS_PER_ROW=4 (96 bytes). */
    float scores_batched[N_POS];
    tq_kv_1b_attention_multi(query, &kv_cache[0], scores_batched, N_POS, HEAD_DIM,
                             BLOCKS_PER_ROW);

    /* --- Path B: per-position reference (seq_len=1, no stride issue) ---
     * For each position, pass a pointer to head 0's blocks with manual stride. */
    float scores_ref[N_POS];
    for (int pos = 0; pos < N_POS; pos++) {
        const block_tq_kv_1b * head0_blocks = &kv_cache[pos * BLOCKS_PER_ROW];
        tq_kv_1b_attention_multi(query, head0_blocks, &scores_ref[pos], 1, HEAD_DIM, 0);
    }

    /* --- Compare --- */
    int n_fail = 0;
    printf("%-5s  %12s  %12s  %8s\n", "pos", "batched", "reference", "match");
    for (int pos = 0; pos < N_POS; pos++) {
        float diff = fabsf(scores_batched[pos] - scores_ref[pos]);
        int ok = (diff < 1e-4f);
        printf("  %3d  %12.6f  %12.6f  %s\n",
               pos, scores_batched[pos], scores_ref[pos],
               ok ? "  OK" : "FAIL");
        if (!ok) n_fail++;
    }

    printf("\n%d/%d positions match.\n", N_POS - n_fail, N_POS);
    if (n_fail > 0) {
        printf("FAIL: K stride bug detected — batched call reads wrong positions.\n");
        return 1;
    }
    printf("PASS: K stride is correct.\n");
    return 0;
}
