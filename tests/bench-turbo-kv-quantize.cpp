/*
 * bench-turbo-kv-quantize.cpp — micro-benchmark for PHASE25 step 3
 * quantize-path profiling. Sibling to bench-turbo-kv-4b-avx2.cpp
 * (which measured the vec_dot hot path).
 *
 * Hammers quantize_row_turbo_kv_4b_ref in a tight loop with minimal
 * setup noise. Every call does: L2 norm + normalize + RHT forward +
 * max_abs + nearest-centroid lookup per block.
 *
 * Usage (Zen 2):
 *   perf stat -e cycles,instructions,branches,branch-misses,\
 *     L1-dcache-loads,L1-dcache-load-misses \
 *     build-tq/bin/bench-turbo-kv-quantize
 *
 *   perf record -F 9999 -e cycles:u --call-graph=fp -o /tmp/bench-q.perf \
 *     build-tq/bin/bench-turbo-kv-quantize
 *   perf report -i /tmp/bench-q.perf --stdio --sort=symbol --no-children
 *
 * Same design rule as the vec_dot bench: keep the inner loop tight
 * and let perf attribute to the real kernel function, not the
 * harness. quantize_row_turbo_kv_4b_ref is exported (non-static), so
 * it shows up as its own symbol — unlike the inlined AVX2 path.
 */

#include "ggml-turbo-kv.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

/* Working set tuned to fit in L2. 2048 source rows × 128 floats × 4B
 * = 1 MB — a bit larger than the vec_dot bench's 144 KB because the
 * source rows themselves are uncompressed. Still fits in Zen 2's
 * 512 KB L2 per-core with room for the output pool + stack. */
static constexpr int N_ROWS      = 2048;
static constexpr int TOTAL_CALLS = 1000000;
static constexpr int HEAD_DIM    = TURBO_KV_BLOCK_SIZE;

int main() {
    std::mt19937 gen(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    /* Pool of random f32 source rows. Contiguous so the hot loop's
     * address arithmetic stays linear. */
    std::vector<float> src_pool(N_ROWS * HEAD_DIM);
    for (int r = 0; r < N_ROWS; r++) {
        float * row = src_pool.data() + r * HEAD_DIM;
        float norm2 = 0.0f;
        for (int i = 0; i < HEAD_DIM; i++) {
            row[i] = dist(gen);
            norm2 += row[i] * row[i];
        }
        /* Spec requires L2_norm > 0. Enforce. */
        if (norm2 < 1e-6f) row[0] = 1.0f;
    }

    /* Single output block, overwritten per call. The output doesn't
     * factor into the kernel's cycle count; the write is one
     * 72-byte store. Cache-local. */
    block_turbo_kv_4b dst;

    /* Warm the cache + any lazy init. */
    quantize_row_turbo_kv_4b_ref(src_pool.data(), &dst, HEAD_DIM);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < TOTAL_CALLS; i++) {
        const int r = i & (N_ROWS - 1);
        quantize_row_turbo_kv_4b_ref(
            src_pool.data() + r * HEAD_DIM, &dst, HEAD_DIM);
    }
    const auto t1 = std::chrono::steady_clock::now();

    /* Anti-dead-code: read one byte from dst so the compiler can't
     * assume the output is unused. */
    volatile uint8_t sink = dst.mse_indices[0];
    (void) sink;

    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    const double ns_per_call = ns / TOTAL_CALLS;
    const double calls_per_sec = 1e9 / ns_per_call;

    fprintf(stderr, "quantize_row_turbo_kv_4b_ref (one block, %d elems):\n", HEAD_DIM);
    fprintf(stderr, "  total calls     : %d\n", TOTAL_CALLS);
    fprintf(stderr, "  working rows    : %d (~%zu bytes)\n",
            N_ROWS, N_ROWS * HEAD_DIM * sizeof(float));
    fprintf(stderr, "  elapsed         : %.3f ms\n", ns / 1e6);
    fprintf(stderr, "  ns per call     : %.2f\n", ns_per_call);
    fprintf(stderr, "  calls per sec   : %.2e\n", calls_per_sec);
    return 0;
}
