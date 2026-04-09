/*
 * test-tq-rht-roundtrip.c — Diagnostic for Suspect #2: RHT seed mismatch
 *
 * Single-block (head_dim=128), 4 known keys, 1 query.
 * Verifies that:
 *   a) stored rht_seed == TQ_DEFAULT_SEED in every block
 *   b) TQ attention score ranking matches FP32 dot-product ranking
 *
 * If the quantization and attention paths use different seeds, the
 * sign hashes will be uncorrelated and scores will be noise — the
 * ranking test catches this.
 *
 * Expected result: PASS (both paths use TQ_DEFAULT_SEED = 0x12345678).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ggml-turbo-quant.h"

#define HEAD_DIM 128
#define N_KEYS   4
#define TQ_DEFAULT_SEED_VALUE 0x12345678u

/* Simple deterministic PRNG. */
static uint32_t xor_state;
static float xrand(void) {
    xor_state ^= xor_state << 13;
    xor_state ^= xor_state >> 17;
    xor_state ^= xor_state << 5;
    return (float)(xor_state & 0xFFFF) / 65536.0f - 0.5f;
}

static int argsort_cmp_desc(const void *a, const void *b) {
    /* Sort indices by descending score. */
    return 0; /* placeholder, we do manual argsort */
}

int main(void) {
    float keys[N_KEYS][HEAD_DIM];
    float query[HEAD_DIM];
    block_tq_kv_1b blocks[N_KEYS];

    /* Generate deterministic keys and query. */
    for (int k = 0; k < N_KEYS; k++) {
        xor_state = 1000 + k * 7;
        for (int i = 0; i < HEAD_DIM; i++) keys[k][i] = xrand();
    }
    xor_state = 9999;
    for (int i = 0; i < HEAD_DIM; i++) query[i] = xrand();

    /* Quantize keys. */
    for (int k = 0; k < N_KEYS; k++) {
        quantize_row_tq_kv_1b_ref(keys[k], &blocks[k], HEAD_DIM);
    }

    /* Check a: stored seed. */
    int seed_ok = 1;
    for (int k = 0; k < N_KEYS; k++) {
        if (blocks[k].rht_seed != TQ_DEFAULT_SEED_VALUE) {
            printf("FAIL: block %d rht_seed = 0x%08X, expected 0x%08X\n",
                   k, blocks[k].rht_seed, TQ_DEFAULT_SEED_VALUE);
            seed_ok = 0;
        }
    }
    if (!seed_ok) return 1;
    printf("Seed check: all %d blocks have rht_seed = 0x%08X. OK.\n",
           N_KEYS, TQ_DEFAULT_SEED_VALUE);

    /* Compute TQ attention scores (single-block variant, no stride issue). */
    float tq_scores[N_KEYS];
    tq_kv_1b_attention(query, blocks, tq_scores, N_KEYS, HEAD_DIM);

    /* Compute FP32 dot products. */
    float fp_dots[N_KEYS];
    for (int k = 0; k < N_KEYS; k++) {
        float dot = 0.0f;
        for (int i = 0; i < HEAD_DIM; i++) dot += query[i] * keys[k][i];
        fp_dots[k] = dot;
    }

    /* Print both. */
    printf("\n%-5s  %12s  %12s\n", "key", "TQ score", "FP32 dot");
    for (int k = 0; k < N_KEYS; k++) {
        printf("  %3d  %12.6f  %12.6f\n", k, tq_scores[k], fp_dots[k]);
    }

    /* Check b: ranking match.
     * Find the argmax of TQ scores and of FP32 dots.
     * They should agree (same key is "most similar"). */
    int tq_argmax = 0, fp_argmax = 0;
    for (int k = 1; k < N_KEYS; k++) {
        if (tq_scores[k] > tq_scores[tq_argmax]) tq_argmax = k;
        if (fp_dots[k]  > fp_dots[fp_argmax])     fp_argmax = k;
    }

    printf("\nArgmax: TQ=%d, FP32=%d\n", tq_argmax, fp_argmax);

    /* Also check sign agreement (all TQ scores should have the same sign
     * as the corresponding FP32 dot product). */
    int sign_ok = 1;
    for (int k = 0; k < N_KEYS; k++) {
        int tq_sign = (tq_scores[k] >= 0) ? 1 : -1;
        int fp_sign = (fp_dots[k]   >= 0) ? 1 : -1;
        if (tq_sign != fp_sign) {
            printf("FAIL: sign mismatch at key %d: TQ=%+d, FP32=%+d\n",
                   k, tq_sign, fp_sign);
            sign_ok = 0;
        }
    }

    if (!sign_ok) {
        printf("FAIL: sign disagreement between TQ and FP32 — possible seed mismatch.\n");
        return 1;
    }

    if (tq_argmax != fp_argmax) {
        printf("FAIL: ranking mismatch — TQ picks key %d, FP32 picks key %d.\n",
               tq_argmax, fp_argmax);
        return 1;
    }

    printf("PASS: RHT seed is consistent; TQ ranking matches FP32.\n");
    return 0;
}
