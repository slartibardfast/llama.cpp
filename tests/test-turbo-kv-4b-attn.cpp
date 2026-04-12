/*
 * test-turbo-kv-4b-attn.cpp — correctness fixture for turbo_kv_4b_attention_multi
 *
 * Generates deterministic Q and K vectors, quantizes K to turbo_kv_4b blocks,
 * computes reference scores via the per-call scalar vec_dot (known correct from
 * 12/12 tool-call battery), then compares against turbo_kv_4b_attention_multi.
 *
 * Validates:
 *   - Single-block (head_dim=128) and multi-block (head_dim=256) rows
 *   - valid_count = 1, 10, 500
 *   - k_stride_blocks = 0 (dense) and k_stride_blocks > n_blocks (GQA interleave)
 *   - Max abs error < 1e-4 between per-call and batched paths
 *
 * Build: cmake --build build-tq --target test-turbo-kv-4b-attn
 * Run:   build-tq/bin/test-turbo-kv-4b-attn
 */

#include "ggml-turbo-kv.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <cassert>

// Deterministic pseudo-random float in [-1, 1]
static float det_rand(uint32_t * state) {
    *state = *state * 1103515245u + 12345u;
    return ((float)(*state >> 16) / 32768.0f) - 1.0f;
}

static bool test_attention_multi(int head_dim, int valid_count, int k_stride_blocks_override) {
    const int n_blocks = (head_dim + TURBO_KV_BLOCK_SIZE - 1) / TURBO_KV_BLOCK_SIZE;
    const int k_stride_blocks = (k_stride_blocks_override > 0)
        ? k_stride_blocks_override
        : n_blocks;

    // Generate deterministic Q
    std::vector<float> q(head_dim);
    uint32_t seed = 42;
    for (int i = 0; i < head_dim; i++) {
        q[i] = det_rand(&seed);
    }

    // Generate deterministic K rows and quantize
    // Each K position has k_stride_blocks blocks (contiguous in memory)
    std::vector<block_turbo_kv_4b> k_cache(valid_count * k_stride_blocks);
    std::vector<float> k_raw(valid_count * head_dim);

    for (int s = 0; s < valid_count; s++) {
        float * k_row = k_raw.data() + s * head_dim;
        for (int i = 0; i < head_dim; i++) {
            k_row[i] = det_rand(&seed);
        }
        // Quantize this row into contiguous blocks at stride position
        block_turbo_kv_4b * dst = k_cache.data() + s * k_stride_blocks;
        quantize_row_turbo_kv_4b_ref(k_row, dst, head_dim);
    }

    // Reference: per-call scalar vec_dot (rotates Q internally each call)
    std::vector<float> ref_scores(valid_count);
    for (int s = 0; s < valid_count; s++) {
        const block_turbo_kv_4b * k_blocks = k_cache.data() + s * k_stride_blocks;
        ggml_vec_dot_turbo_kv_4b_f32(
            head_dim, &ref_scores[s], 0,
            k_blocks, 0,
            q.data(), 0,
            1);
    }

    // Test: batched attention_multi (rotates Q once, loops K)
    std::vector<float> test_scores(valid_count, 0.0f);

    // turbo_kv_4b_attention_multi is now implemented — test batched vs per-call.
    turbo_kv_4b_attention_multi(
        q.data(),
        k_cache.data(),
        test_scores.data(),
        valid_count,
        head_dim,
        k_stride_blocks);

    // Compare
    float max_err = 0.0f;
    int max_err_idx = -1;
    for (int s = 0; s < valid_count; s++) {
        float err = fabsf(ref_scores[s] - test_scores[s]);
        if (err > max_err) {
            max_err = err;
            max_err_idx = s;
        }
    }

    const float tolerance = 1e-4f;
    bool pass = (max_err < tolerance);

    fprintf(stderr, "  head_dim=%3d valid=%4d stride=%2d | max_err=%.6f at [%d] | %s\n",
            head_dim, valid_count, k_stride_blocks,
            max_err, max_err_idx, pass ? "PASS" : "FAIL");

    if (!pass) {
        fprintf(stderr, "    ref[%d]=%.6f test[%d]=%.6f\n",
                max_err_idx, ref_scores[max_err_idx],
                max_err_idx, test_scores[max_err_idx]);
    }

    return pass;
}

int main() {
    fprintf(stderr, "=== turbo_kv_4b attention correctness ===\n");

    int pass = 0, fail = 0;

    // Single block (head_dim=128)
    test_attention_multi(128,   1, 0) ? pass++ : fail++;
    test_attention_multi(128,  10, 0) ? pass++ : fail++;
    test_attention_multi(128, 500, 0) ? pass++ : fail++;

    // Multi-block (head_dim=256, Qwen3.5 config)
    test_attention_multi(256,   1, 0) ? pass++ : fail++;
    test_attention_multi(256,  10, 0) ? pass++ : fail++;
    test_attention_multi(256, 500, 0) ? pass++ : fail++;

    // GQA interleave (stride > n_blocks)
    test_attention_multi(256,  10, 4) ? pass++ : fail++;
    test_attention_multi(128,  10, 4) ? pass++ : fail++;

    fprintf(stderr, "\n=== %d/%d passed ===\n", pass, pass + fail);
    return fail > 0 ? 1 : 0;
}
