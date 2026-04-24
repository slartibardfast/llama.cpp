/*
 * bench-turbo-kv-4b-avx2.cpp — micro-benchmark for PHASE25 step 3 profiling.
 *
 * Hammers turbo_kv_4b_avx2_single_block_dot in a tight loop with minimal
 * setup/teardown noise, so perf-stat counter attribution lands on the
 * AVX2 inner kernel's hot instructions rather than the harness.
 *
 * Workload:
 *   - N_BLOCKS quantized K blocks (pre-rotated random data, cached).
 *   - 1 pre-rotated query of head_dim=128 elements.
 *   - Inner loop calls single_block_dot across all N_BLOCKS each
 *     iteration, for TOTAL_CALLS total calls.
 *
 * Usage:
 *   # Hot-instruction attribution — needs perf to sample the kernel:
 *   perf record -F 999 -g -- build-tq/bin/bench-turbo-kv-4b-avx2
 *   perf report --stdio --call-graph=none | head -30
 *
 *   # Top-down perf-stat for IPC / retired ops per kernel call:
 *   perf stat -e cycles,instructions,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses \
 *       build-tq/bin/bench-turbo-kv-4b-avx2
 */

#include "ggml-turbo-kv.h"
#include "ggml-cpu/arch/x86/turbo_kv_4b_avx2.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifndef GGML_TURBO_KV_4B_HAVE_AVX2
#  error "This bench requires AVX2; build with -march=native on Zen 2+."
#endif

/* Keep this tight — we want the kernel to dominate, not the harness. */
static constexpr int N_BLOCKS    = 2048;   /* working set: ~144 KB (fits in L2) */
static constexpr int TOTAL_CALLS = 1000000;
static constexpr int HEAD_DIM    = TURBO_KV_BLOCK_SIZE;

int main() {
    std::mt19937 gen(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    /* Build a pool of quantized K blocks. */
    std::vector<block_turbo_kv_4b> blocks(N_BLOCKS);
    std::vector<float> raw(HEAD_DIM);
    for (int b = 0; b < N_BLOCKS; b++) {
        for (int i = 0; i < HEAD_DIM; i++) raw[i] = dist(gen);
        /* Ensure non-degenerate norm (spec precondition). */
        float norm2 = 0;
        for (int i = 0; i < HEAD_DIM; i++) norm2 += raw[i] * raw[i];
        if (norm2 < 1e-6f) raw[0] = 1.0f;
        quantize_row_turbo_kv_4b_ref(raw.data(), &blocks[b], HEAD_DIM);
    }

    /* One pre-rotated query. The _cpu dispatch pre-rotates per call in
     * production; here we hoist it so perf attributes to single_block_dot
     * not rotate_query. */
    std::vector<float> q_raw(HEAD_DIM), q_rot(HEAD_DIM);
    for (int i = 0; i < HEAD_DIM; i++) q_raw[i] = dist(gen);
    memcpy(q_rot.data(), q_raw.data(), HEAD_DIM * sizeof(float));
    turbo_kv_rht_forward(q_rot.data(), HEAD_DIM, TURBO_KV_DEFAULT_SEED);

    /* Warm the codebook LUT init. */
    volatile float sink = 0.0f;
    sink += turbo_kv_4b_avx2_single_block_dot(&blocks[0], q_rot.data(), HEAD_DIM);

    /* Hot loop: round-robin through the pool so branch predictors don't
     * degenerate to trivial patterns. */
    const auto t0 = std::chrono::steady_clock::now();
    float total = 0.0f;
    for (int i = 0; i < TOTAL_CALLS; i++) {
        const int b = i & (N_BLOCKS - 1);
        total += turbo_kv_4b_avx2_single_block_dot(
            &blocks[b], q_rot.data(), HEAD_DIM);
    }
    const auto t1 = std::chrono::steady_clock::now();
    sink += total;

    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    const double ns_per_call = ns / TOTAL_CALLS;
    const double calls_per_sec = 1e9 / ns_per_call;

    fprintf(stderr, "turbo_kv_4b_avx2_single_block_dot:\n");
    fprintf(stderr, "  total calls     : %d\n", TOTAL_CALLS);
    fprintf(stderr, "  working blocks  : %d (~%zu bytes)\n",
            N_BLOCKS, N_BLOCKS * sizeof(block_turbo_kv_4b));
    fprintf(stderr, "  elapsed         : %.3f ms\n", ns / 1e6);
    fprintf(stderr, "  ns per call     : %.2f\n", ns_per_call);
    fprintf(stderr, "  calls per sec   : %.2e\n", calls_per_sec);
    fprintf(stderr, "  sink            : %.3f (anti-dead-code)\n", (float) sink);

    return 0;
}
