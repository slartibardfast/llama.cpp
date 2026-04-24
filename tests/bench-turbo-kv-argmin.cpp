/*
 * bench-turbo-kv-argmin.cpp — micro-bench comparing the scalar and
 * AVX2 argmin paths for the quantize Step 5 hot loop, isolated from
 * the surrounding L2-norm / normalize / RHT / max_abs steps.
 *
 * Runs each path for N iterations on the same pre-computed
 * (rotated, inv_std) pool, reports ns/call and the speedup ratio.
 *
 * The bit-exactness contract is enforced separately by
 * property_AVX2_NearestCentroidAssignment_matches_reference in
 * test-turbo-kv-pbt.cpp; this file is only for throughput.
 */

#include "ggml-turbo-kv.h"
#include "ggml-cpu/arch/x86/turbo_kv_4b_avx2.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#ifndef GGML_TURBO_KV_4B_HAVE_AVX2
#  error "This bench requires AVX2; build with -march=native."
#endif

static constexpr int N_POOL     = 512;
static constexpr int TOTAL      = 500000;
static constexpr int HEAD_DIM   = TURBO_KV_BLOCK_SIZE;

/* Scalar reference: the same inner loop as quantize_block_turbo_kv_4b's
 * Step 5, isolated. */
static void scalar_argmin_block(const float * rotated, float inv_std,
                                 uint8_t * out_bytes)
{
    memset(out_bytes, 0, HEAD_DIM / 2);
    for (int i = 0; i < HEAD_DIM; i++) {
        const float x = rotated[i] * inv_std;
        int best = 0;
        float best_dist = fabsf(x - turbo_kv_4b_codebook[0]);
        for (int c = 1; c < 16; c++) {
            const float d = fabsf(x - turbo_kv_4b_codebook[c]);
            if (d < best_dist) { best_dist = d; best = c; }
        }
        const int byte_idx = i / 2;
        const int bit_pos  = (i & 1) * 4;
        out_bytes[byte_idx] |= (uint8_t) ((best & 0x0F) << bit_pos);
    }
}

int main() {
    /* Pool of (rotated, inv_std) pairs computed once. Feeds both
     * paths, keeps the argmin in isolation. */
    std::mt19937 gen(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> rotated_pool(N_POOL * HEAD_DIM);
    std::vector<float> inv_std_pool(N_POOL);

    for (int r = 0; r < N_POOL; r++) {
        float * row = rotated_pool.data() + r * HEAD_DIM;
        float norm2 = 0;
        for (int i = 0; i < HEAD_DIM; i++) {
            row[i] = dist(gen);
            norm2 += row[i] * row[i];
        }
        const float norm = sqrtf(norm2);
        const float inv_norm = (norm > 1e-6f) ? (1.0f / norm) : 1.0f;
        for (int i = 0; i < HEAD_DIM; i++) row[i] *= inv_norm;
        turbo_kv_rht_forward(row, HEAD_DIM, TURBO_KV_DEFAULT_SEED);
        float max_abs = 0;
        for (int i = 0; i < HEAD_DIM; i++) {
            const float a = fabsf(row[i]);
            if (a > max_abs) max_abs = a;
        }
        if (max_abs < 1e-10f) max_abs = 1.0f;
        inv_std_pool[r] = TURBO_KV_4B_CENT_MAX / max_abs;
    }

    uint8_t out[HEAD_DIM / 2];
    volatile uint8_t sink = 0;

    /* Warm-up. */
    scalar_argmin_block(rotated_pool.data(), inv_std_pool[0], out);
    turbo_kv_4b_avx2_nearest_centroid_block(rotated_pool.data(), inv_std_pool[0], out);

    /* Scalar timing. */
    const auto s0 = std::chrono::steady_clock::now();
    for (int n = 0; n < TOTAL; n++) {
        const int r = n & (N_POOL - 1);
        scalar_argmin_block(
            rotated_pool.data() + r * HEAD_DIM,
            inv_std_pool[r],
            out);
        sink ^= out[0];
    }
    const auto s1 = std::chrono::steady_clock::now();
    const double scalar_ns = std::chrono::duration<double, std::nano>(s1 - s0).count() / TOTAL;

    /* AVX2 timing. */
    const auto a0 = std::chrono::steady_clock::now();
    for (int n = 0; n < TOTAL; n++) {
        const int r = n & (N_POOL - 1);
        turbo_kv_4b_avx2_nearest_centroid_block(
            rotated_pool.data() + r * HEAD_DIM,
            inv_std_pool[r],
            out);
        sink ^= out[0];
    }
    const auto a1 = std::chrono::steady_clock::now();
    const double avx2_ns = std::chrono::duration<double, std::nano>(a1 - a0).count() / TOTAL;

    fprintf(stderr, "turbo_kv_4b argmin Step 5 (one block, %d elems):\n", HEAD_DIM);
    fprintf(stderr, "  scalar ns/call : %7.2f\n", scalar_ns);
    fprintf(stderr, "  AVX2   ns/call : %7.2f\n", avx2_ns);
    fprintf(stderr, "  speedup        : %.2fx\n", scalar_ns / avx2_ns);
    fprintf(stderr, "  (sink: %u)\n", (unsigned) sink);
    return 0;
}
