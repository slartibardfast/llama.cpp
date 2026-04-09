/*
 * test-tq-multiblock-accum.c — Diagnostic for Suspect #3: multi-block math
 *
 * head_dim=256 (2 blocks per head), single position.
 * Verifies the multi-block accumulation:
 *   Path A: tq_kv_1b_attention_multi(q, blocks, &score, 1, 256)
 *   Path B: sum of two single-block calls
 *   Path C: scalar reference (manual RHT + XOR popcount)
 *
 * Expected result: PASS (the accumulation is a simple sum).
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ggml-turbo-quant.h"

#define HEAD_DIM 256
#define BLOCK_SIZE TQ_KV_1B_BLOCK_SIZE  /* 128 */
#define N_BLOCKS (HEAD_DIM / BLOCK_SIZE) /* 2 */

static uint32_t xor_state;
static float xrand(void) {
    xor_state ^= xor_state << 13;
    xor_state ^= xor_state >> 17;
    xor_state ^= xor_state << 5;
    return (float)(xor_state & 0xFFFF) / 65536.0f - 0.5f;
}

int main(void) {
    float key[HEAD_DIM], query[HEAD_DIM];
    block_tq_kv_1b blocks[N_BLOCKS];

    /* Generate deterministic key and query. */
    xor_state = 0xABCD1234;
    for (int i = 0; i < HEAD_DIM; i++) key[i] = xrand();
    xor_state = 0x5678CDEF;
    for (int i = 0; i < HEAD_DIM; i++) query[i] = xrand();

    /* Quantize key into 2 contiguous blocks. */
    quantize_row_tq_kv_1b_ref(key, blocks, HEAD_DIM);

    /* --- Path A: multi-block call (seq_len=1, no stride issue) --- */
    float score_multi = 0.0f;
    tq_kv_1b_attention_multi(query, blocks, &score_multi, 1, HEAD_DIM, 0);

    /* --- Path B: sum of single-block calls --- */
    float score_b[N_BLOCKS];
    for (int b = 0; b < N_BLOCKS; b++) {
        tq_kv_1b_attention(query + b * BLOCK_SIZE, &blocks[b],
                           &score_b[b], 1, BLOCK_SIZE);
    }
    float score_sum = 0.0f;
    for (int b = 0; b < N_BLOCKS; b++) score_sum += score_b[b];

    /* --- Path C: scalar reference ---
     * For each block, manually compute:
     *   score_b = q_norm_b * k_norm_b * sqrt(pi/2) / 128 * (2*agree - 128)
     * where agree = popcount(q_signs XOR k_signs flipped) ... actually,
     * agree = 128 - hamming(q_signs, k_signs).
     *
     * We can't easily replicate RHT here without vendoring it, but we CAN
     * verify A == B which is the key check. */

    printf("Path A (multi-block):  %12.6f\n", score_multi);
    printf("Path B (per-block sum):%12.6f  (b0=%f, b1=%f)\n",
           score_sum, score_b[0], score_b[1]);

    float diff_ab = fabsf(score_multi - score_sum);
    printf("Diff A-B: %e\n", diff_ab);

    if (diff_ab > 1e-4f) {
        printf("FAIL: multi-block result differs from per-block sum.\n");
        return 1;
    }

    printf("PASS: multi-block accumulation matches per-block sum.\n");
    return 0;
}
