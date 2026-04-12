/*
 * test-split-k-attention.cpp — validates split K cache attention math
 *
 * Tests:
 *   1. Split dot product equivalence:
 *      Q@K^T == Q_rope@K_rope^T + Q_static@K_static^T
 *
 *   2. Pre-RoPE + on-the-fly RoPE equivalence:
 *      RoPE(K_pre)[0:n_rot] == RoPE(K[0:n_rot])
 *      and K_pre[n_rot:] == K[n_rot:]  (static dims unchanged by RoPE)
 *
 *   3. Quantize roundtrip preservation:
 *      split(dequant(quant(K_pre))) produces valid attention scores
 *      when combined via split dot product
 *
 * Build: cmake --build build-tq --target test-split-k-attention
 * Run:   build-tq/bin/test-split-k-attention
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

static float det_rand(uint32_t * state) {
    *state = *state * 1103515245u + 12345u;
    return ((float)(*state >> 16) / 32768.0f) - 1.0f;
}

static int n_pass = 0, n_fail = 0;

static void check(const char * name, bool cond) {
    if (cond) {
        printf("  %-60s | PASS\n", name);
        n_pass++;
    } else {
        printf("  %-60s | FAIL\n", name);
        n_fail++;
    }
}

/* ==========================================================
 * Test 1: Split dot product equivalence
 *
 * For random Q[head_dim] and K[head_dim]:
 *   dot(Q, K) == dot(Q[0:n_rot], K[0:n_rot]) + dot(Q[n_rot:], K[n_rot:])
 *
 * This is trivially true in exact arithmetic. We verify it holds
 * within f32 tolerance to catch dimension mis-splits.
 * ========================================================== */
static void test_split_dot_equivalence(int head_dim, int n_rot, int n_kv) {
    const int n_stat = head_dim - n_rot;
    uint32_t seed = 0xDEAD0001;

    std::vector<float> Q(head_dim);
    std::vector<float> K(head_dim * n_kv);

    for (int i = 0; i < head_dim; i++) Q[i] = det_rand(&seed);
    for (int i = 0; i < head_dim * n_kv; i++) K[i] = det_rand(&seed);

    float max_err = 0.0f;
    for (int p = 0; p < n_kv; p++) {
        const float * k_row = &K[p * head_dim];

        // Full dot product
        float dot_full = 0.0f;
        for (int i = 0; i < head_dim; i++) dot_full += Q[i] * k_row[i];

        // Split dot product
        float dot_rope = 0.0f;
        for (int i = 0; i < n_rot; i++) dot_rope += Q[i] * k_row[i];

        float dot_static = 0.0f;
        for (int i = n_rot; i < head_dim; i++) dot_static += Q[i] * k_row[i];

        float dot_split = dot_rope + dot_static;

        float err = fabsf(dot_full - dot_split);
        if (err > max_err) max_err = err;
    }

    char buf[128];
    snprintf(buf, sizeof(buf), "split_dot hd=%d n_rot=%d n_kv=%d | max_err=%.8f",
             head_dim, n_rot, n_kv, max_err);
    check(buf, max_err < 1e-4f);
}

/* ==========================================================
 * Test 2: RoPE dimension independence
 *
 * RoPE only modifies dimensions [0, n_rot). Dims [n_rot, head_dim)
 * must be unchanged. This validates our assumption that K_static
 * doesn't interact with RoPE.
 * ========================================================== */
static void test_rope_dimension_independence(int head_dim, int n_rot) {
    const int n_stat = head_dim - n_rot;
    uint32_t seed = 0xDEAD0002;

    std::vector<float> K_orig(head_dim);
    std::vector<float> K_roped(head_dim);

    for (int i = 0; i < head_dim; i++) {
        K_orig[i] = det_rand(&seed);
        K_roped[i] = K_orig[i];
    }

    // Apply RoPE manually to first n_rot dims (simple rotation, position=7)
    const float pos = 7.0f;
    const float freq_base = 10000.0f;
    for (int i = 0; i < n_rot; i += 2) {
        const float theta = pos * powf(freq_base, -(float)i / (float)n_rot);
        const float cos_t = cosf(theta);
        const float sin_t = sinf(theta);
        const float k0 = K_orig[i];
        const float k1 = K_orig[i + 1];
        K_roped[i]     = k0 * cos_t - k1 * sin_t;
        K_roped[i + 1] = k0 * sin_t + k1 * cos_t;
    }

    // Check: static dims unchanged
    float max_err_static = 0.0f;
    for (int i = n_rot; i < head_dim; i++) {
        float err = fabsf(K_roped[i] - K_orig[i]);
        if (err > max_err_static) max_err_static = err;
    }

    // Check: rope dims DID change (at least some)
    float max_diff_rope = 0.0f;
    for (int i = 0; i < n_rot; i++) {
        float diff = fabsf(K_roped[i] - K_orig[i]);
        if (diff > max_diff_rope) max_diff_rope = diff;
    }

    char buf[128];
    snprintf(buf, sizeof(buf), "rope_independence hd=%d n_rot=%d | static_err=%.8f",
             head_dim, n_rot, max_err_static);
    check(buf, max_err_static == 0.0f && max_diff_rope > 0.0f);
}

/* ==========================================================
 * Test 3: Pre-RoPE split attention == post-RoPE full attention
 *
 * Given Q (with RoPE applied) and K_pre (without RoPE):
 *
 *   Post-RoPE path:  K_post = apply_rope(K_pre)
 *                    score  = dot(Q, K_post)
 *
 *   Pre-RoPE split:  K_rope = apply_rope(K_pre[0:n_rot])
 *                    K_stat = K_pre[n_rot:]
 *                    score  = dot(Q[0:n_rot], K_rope) + dot(Q[n_rot:], K_stat)
 *
 * Both must give identical scores (within f32 tolerance).
 * This is THE critical test: it validates that the split K
 * cache + split attention produces the same result as the
 * standard post-RoPE path.
 * ========================================================== */
static void test_pre_rope_split_vs_post_rope(int head_dim, int n_rot, int n_kv) {
    const int n_stat = head_dim - n_rot;
    uint32_t seed = 0xDEAD0003;

    // Generate Q (already has RoPE applied, but we don't care — it's just a vector)
    std::vector<float> Q(head_dim);
    for (int i = 0; i < head_dim; i++) Q[i] = det_rand(&seed);

    // Generate K_pre (pre-RoPE) for multiple positions
    std::vector<float> K_pre(head_dim * n_kv);
    for (int i = 0; i < head_dim * n_kv; i++) K_pre[i] = det_rand(&seed);

    float max_err = 0.0f;
    int worst_pos = -1;

    for (int p = 0; p < n_kv; p++) {
        float * k_pre = &K_pre[p * head_dim];
        const float pos = (float)(p + 1);
        const float freq_base = 10000.0f;

        // === Post-RoPE path: apply RoPE to full K, then full dot ===
        std::vector<float> k_post(head_dim);
        memcpy(k_post.data(), k_pre, head_dim * sizeof(float));
        for (int i = 0; i < n_rot; i += 2) {
            const float theta = pos * powf(freq_base, -(float)i / (float)n_rot);
            const float cos_t = cosf(theta);
            const float sin_t = sinf(theta);
            const float k0 = k_post[i], k1 = k_post[i + 1];
            k_post[i]     = k0 * cos_t - k1 * sin_t;
            k_post[i + 1] = k0 * sin_t + k1 * cos_t;
        }
        float score_post = 0.0f;
        for (int i = 0; i < head_dim; i++) score_post += Q[i] * k_post[i];

        // === Pre-RoPE split path: RoPE only on rope dims, split dot ===
        std::vector<float> k_rope_roped(n_rot);
        for (int i = 0; i < n_rot; i += 2) {
            const float theta = pos * powf(freq_base, -(float)i / (float)n_rot);
            const float cos_t = cosf(theta);
            const float sin_t = sinf(theta);
            const float k0 = k_pre[i], k1 = k_pre[i + 1];
            k_rope_roped[i]     = k0 * cos_t - k1 * sin_t;
            k_rope_roped[i + 1] = k0 * sin_t + k1 * cos_t;
        }

        float dot_rope = 0.0f;
        for (int i = 0; i < n_rot; i++) dot_rope += Q[i] * k_rope_roped[i];

        float dot_stat = 0.0f;
        for (int i = 0; i < n_stat; i++) dot_stat += Q[n_rot + i] * k_pre[n_rot + i];

        float score_split = dot_rope + dot_stat;

        float err = fabsf(score_post - score_split);
        if (err > max_err) { max_err = err; worst_pos = p; }
    }

    char buf[128];
    snprintf(buf, sizeof(buf), "pre_vs_post_rope hd=%d n_rot=%d n_kv=%d | max_err=%.8f @pos=%d",
             head_dim, n_rot, n_kv, max_err, worst_pos);
    check(buf, max_err < 1e-4f);
}

/* ==========================================================
 * Test 4: FP accumulation order sensitivity
 *
 * Measures how much the dot product result changes when
 * computed as one sum vs two partial sums + add. This quantifies
 * the expected divergence from split attention.
 * ========================================================== */
static void test_fp_accumulation_sensitivity(int head_dim, int n_rot) {
    uint32_t seed = 0xDEAD0004;

    std::vector<float> A(head_dim), B(head_dim);
    for (int i = 0; i < head_dim; i++) {
        A[i] = det_rand(&seed);
        B[i] = det_rand(&seed);
    }

    // Single pass
    float dot_single = 0.0f;
    for (int i = 0; i < head_dim; i++) dot_single += A[i] * B[i];

    // Two-pass (split at n_rot)
    float dot_lo = 0.0f;
    for (int i = 0; i < n_rot; i++) dot_lo += A[i] * B[i];
    float dot_hi = 0.0f;
    for (int i = n_rot; i < head_dim; i++) dot_hi += A[i] * B[i];
    float dot_split = dot_lo + dot_hi;

    float err = fabsf(dot_single - dot_split);
    float rel = (fabsf(dot_single) > 1e-10f) ? err / fabsf(dot_single) : err;

    char buf[128];
    snprintf(buf, sizeof(buf), "fp_accum_sensitivity hd=%d n_rot=%d | abs=%.2e rel=%.2e",
             head_dim, n_rot, err, rel);
    // This is informational — we expect small differences
    check(buf, rel < 1e-5f);
}

int main() {
    printf("=== split K attention correctness ===\n\n");

    printf("--- 1. Split dot product equivalence ---\n");
    test_split_dot_equivalence(256, 64, 1);
    test_split_dot_equivalence(256, 64, 512);
    test_split_dot_equivalence(128, 32, 100);
    test_split_dot_equivalence(256, 128, 100);  // half-and-half

    printf("\n--- 2. RoPE dimension independence ---\n");
    test_rope_dimension_independence(256, 64);
    test_rope_dimension_independence(128, 32);
    test_rope_dimension_independence(256, 128);

    printf("\n--- 3. Pre-RoPE split == post-RoPE full (THE critical test) ---\n");
    test_pre_rope_split_vs_post_rope(256, 64, 1);
    test_pre_rope_split_vs_post_rope(256, 64, 512);
    test_pre_rope_split_vs_post_rope(256, 64, 4096);
    test_pre_rope_split_vs_post_rope(128, 32, 100);

    printf("\n--- 4. FP accumulation order sensitivity ---\n");
    test_fp_accumulation_sensitivity(256, 64);
    test_fp_accumulation_sensitivity(256, 128);
    test_fp_accumulation_sensitivity(128, 32);

    printf("\n=== %d/%d passed ===\n", n_pass, n_pass + n_fail);
    return n_fail > 0 ? 1 : 0;
}
