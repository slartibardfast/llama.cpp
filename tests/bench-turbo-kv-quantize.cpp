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
#include "ggml-cpu.h"

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

    /* The production AVX2-aware quantize is a static function in
     * ggml-cpu.c; reach it via the type_traits_cpu dispatch table
     * which is what ggml's mul_mat / cpy / set_rows paths use too. */
    const struct ggml_type_traits_cpu * tt =
        ggml_get_type_traits_cpu(GGML_TYPE_TURBO_KV_4B);
    if (tt == NULL || tt->from_float == NULL) {
        fprintf(stderr, "failed to get type_traits_cpu for TURBO_KV_4B\n");
        return 1;
    }

    /* Warm the cache + any lazy init on both paths. */
    quantize_row_turbo_kv_4b_ref(src_pool.data(), &dst, HEAD_DIM);
    tt->from_float(src_pool.data(), &dst, HEAD_DIM);

    /* Scalar path (ggml-base reference). */
    const auto s0 = std::chrono::steady_clock::now();
    for (int i = 0; i < TOTAL_CALLS; i++) {
        const int r = i & (N_ROWS - 1);
        quantize_row_turbo_kv_4b_ref(
            src_pool.data() + r * HEAD_DIM, &dst, HEAD_DIM);
    }
    const auto s1 = std::chrono::steady_clock::now();
    volatile uint8_t sink_s = dst.mse_indices[0];

    /* AVX2-aware path (ggml-cpu type trait, what production uses). */
    const auto a0 = std::chrono::steady_clock::now();
    for (int i = 0; i < TOTAL_CALLS; i++) {
        const int r = i & (N_ROWS - 1);
        tt->from_float(src_pool.data() + r * HEAD_DIM, &dst, HEAD_DIM);
    }
    const auto a1 = std::chrono::steady_clock::now();
    volatile uint8_t sink_a = dst.mse_indices[0];
    (void) sink_s; (void) sink_a;

    const double scalar_ns = std::chrono::duration<double, std::nano>(s1 - s0).count() / TOTAL_CALLS;
    const double avx2_ns   = std::chrono::duration<double, std::nano>(a1 - a0).count() / TOTAL_CALLS;

    fprintf(stderr, "quantize one block (%d elems, full pipeline — steps 1-5):\n", HEAD_DIM);
    fprintf(stderr, "  scalar (ggml-base _ref)       : %7.2f ns/call\n", scalar_ns);
    fprintf(stderr, "  AVX2   (ggml-cpu production)  : %7.2f ns/call\n", avx2_ns);
    fprintf(stderr, "  end-to-end speedup            : %.2fx\n", scalar_ns / avx2_ns);
    return 0;
}
