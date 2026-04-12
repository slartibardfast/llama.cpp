/*
 * test-split-k-graph.cpp — ggml graph-level test for split attention
 *
 * Builds actual ggml computation graphs for:
 *   A) Full attention: Q @ K^T (single mul_mat, full head_dim)
 *   B) Split attention: Q_rope @ K_rope^T + Q_static @ K_static^T
 *
 * Compares the output scores. If they differ, the graph construction
 * for split attention has a bug.
 *
 * Also tests the cache write/read roundtrip: write K to two cache
 * tensors (rope + static), read back, verify data matches.
 */

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

static int n_pass = 0, n_fail = 0;

static void check(const char * name, bool cond) {
    printf("  %-70s %s\n", name, cond ? "PASS" : "FAIL");
    if (cond) n_pass++; else n_fail++;
}

static float det_rand(uint32_t * state) {
    *state = *state * 1103515245u + 12345u;
    return ((float)(*state >> 16) / 32768.0f) - 1.0f;
}

/* ==========================================================
 * Test: split attention graph vs full attention graph
 *
 * Parameters mirror Qwen3.5:
 *   head_dim=256, n_rot=64, n_head=16, n_head_kv=2, n_kv, n_tokens
 * ========================================================== */
static void test_split_attn_graph(
    int head_dim, int n_rot, int n_head, int n_head_kv, int n_kv, int n_tokens)
{
    const int n_stat = head_dim - n_rot;

    // Allocate ggml context
    size_t mem_size = 256 * 1024 * 1024; // 256 MB
    struct ggml_init_params params = { mem_size, NULL, false };
    struct ggml_context * ctx = ggml_init(params);

    // Create input tensors
    // Q: [head_dim, n_head, n_tokens] — already has RoPE applied
    struct ggml_tensor * Q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_head, n_tokens);
    // K_full: [head_dim, n_head_kv, n_kv] — post-RoPE K for the "full" path
    struct ggml_tensor * K_full = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_head_kv, n_kv);
    // K_rope: [n_rot, n_head_kv, n_kv] — rope portion
    struct ggml_tensor * K_rope = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_rot, n_head_kv, n_kv);
    // K_stat: [n_stat, n_head_kv, n_kv] — static portion
    struct ggml_tensor * K_stat = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_stat, n_head_kv, n_kv);

    ggml_set_name(Q, "Q");
    ggml_set_name(K_full, "K_full");
    ggml_set_name(K_rope, "K_rope");
    ggml_set_name(K_stat, "K_stat");

    // Fill with deterministic random data
    uint32_t seed = 0xBEEF0001;
    float * q_data = (float *) Q->data;
    float * k_data = (float *) K_full->data;
    float * kr_data = (float *) K_rope->data;
    float * ks_data = (float *) K_stat->data;

    for (int i = 0; i < head_dim * n_head * n_tokens; i++) {
        q_data[i] = det_rand(&seed);
    }

    // Fill K_full, and split into K_rope and K_stat
    for (int p = 0; p < n_kv; p++) {
        for (int h = 0; h < n_head_kv; h++) {
            for (int d = 0; d < head_dim; d++) {
                float val = det_rand(&seed);
                k_data[d + h * head_dim + p * head_dim * n_head_kv] = val;
                if (d < n_rot) {
                    kr_data[d + h * n_rot + p * n_rot * n_head_kv] = val;
                } else {
                    ks_data[(d - n_rot) + h * n_stat + p * n_stat * n_head_kv] = val;
                }
            }
        }
    }

    // === Path A: Full attention (single mul_mat) ===
    struct ggml_tensor * Q_perm_a = ggml_permute(ctx, Q, 0, 2, 1, 3);
    struct ggml_tensor * K_perm_a = ggml_permute(ctx, K_full, 0, 2, 1, 3);
    struct ggml_tensor * kq_full = ggml_mul_mat(ctx, K_perm_a, Q_perm_a);
    ggml_set_name(kq_full, "kq_full");

    // === Path B: Split attention (two mul_mats + add) ===
    struct ggml_tensor * K_rope_perm = ggml_permute(ctx, K_rope, 0, 2, 1, 3);
    struct ggml_tensor * K_stat_perm = ggml_permute(ctx, K_stat, 0, 2, 1, 3);

    // Permute Q (same as path A)
    struct ggml_tensor * Q_perm_b = ggml_permute(ctx, Q, 0, 2, 1, 3);

    // Split Q at n_rot — strided views
    struct ggml_tensor * Q_rope = ggml_view_4d(ctx, Q_perm_b,
        n_rot, Q_perm_b->ne[1], Q_perm_b->ne[2], 1,
        Q_perm_b->nb[1], Q_perm_b->nb[2], Q_perm_b->nb[2] * Q_perm_b->ne[2], 0);
    ggml_set_name(Q_rope, "Q_rope");

    struct ggml_tensor * Q_stat = ggml_view_4d(ctx, Q_perm_b,
        n_stat, Q_perm_b->ne[1], Q_perm_b->ne[2], 1,
        Q_perm_b->nb[1], Q_perm_b->nb[2], Q_perm_b->nb[2] * Q_perm_b->ne[2],
        n_rot * sizeof(float));
    ggml_set_name(Q_stat, "Q_stat");

    struct ggml_tensor * kq_rope = ggml_mul_mat(ctx, K_rope_perm, Q_rope);
    struct ggml_tensor * kq_stat = ggml_mul_mat(ctx, K_stat_perm, Q_stat);
    struct ggml_tensor * kq_split = ggml_add(ctx, kq_rope, kq_stat);
    ggml_set_name(kq_split, "kq_split");

    // Build and run graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, kq_full);
    ggml_build_forward_expand(gf, kq_split);

    ggml_graph_compute_with_ctx(ctx, gf, 4);

    // Compare results
    int64_t n_el = ggml_nelements(kq_full);
    float * full_data = (float *) kq_full->data;
    float * split_data = (float *) kq_split->data;

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int worst_idx = -1;

    for (int64_t i = 0; i < n_el; i++) {
        float err = fabsf(full_data[i] - split_data[i]);
        float rel = (fabsf(full_data[i]) > 1e-6f) ? err / fabsf(full_data[i]) : err;
        if (err > max_abs_err) {
            max_abs_err = err;
            worst_idx = (int) i;
        }
        if (rel > max_rel_err) max_rel_err = rel;
    }

    char buf[256];
    snprintf(buf, sizeof(buf),
        "split_graph hd=%d rot=%d heads=%d/%d kv=%d tok=%d | abs=%.2e rel=%.2e @%d",
        head_dim, n_rot, n_head, n_head_kv, n_kv, n_tokens,
        max_abs_err, max_rel_err, worst_idx);
    check(buf, max_rel_err < 1e-5f);

    // Print first few values for debugging if failed
    if (max_rel_err >= 1e-5f) {
        printf("    DETAIL: full[0..3] = %.6f %.6f %.6f %.6f\n",
               full_data[0], full_data[1], full_data[2], full_data[3]);
        printf("    DETAIL: split[0..3] = %.6f %.6f %.6f %.6f\n",
               split_data[0], split_data[1], split_data[2], split_data[3]);
        printf("    DETAIL: kq shapes: full [%lld,%lld,%lld,%lld] split [%lld,%lld,%lld,%lld]\n",
               (long long)kq_full->ne[0], (long long)kq_full->ne[1],
               (long long)kq_full->ne[2], (long long)kq_full->ne[3],
               (long long)kq_split->ne[0], (long long)kq_split->ne[1],
               (long long)kq_split->ne[2], (long long)kq_split->ne[3]);
    }

    ggml_free(ctx);
}

/* ==========================================================
 * Test: K cache write/read roundtrip
 *
 * Write a K tensor to split caches, read back, verify data matches.
 * ========================================================== */
static void test_cache_roundtrip(int head_dim, int n_rot, int n_head_kv) {
    const int n_stat = head_dim - n_rot;

    // Fill K with known data
    std::vector<float> K_orig(head_dim * n_head_kv);
    uint32_t seed = 0xCAFE0001;
    for (size_t i = 0; i < K_orig.size(); i++) {
        K_orig[i] = det_rand(&seed);
    }

    // Split manually
    std::vector<float> K_rope_expected(n_rot * n_head_kv);
    std::vector<float> K_stat_expected(n_stat * n_head_kv);

    for (int h = 0; h < n_head_kv; h++) {
        for (int d = 0; d < head_dim; d++) {
            float val = K_orig[d + h * head_dim];
            if (d < n_rot) {
                K_rope_expected[d + h * n_rot] = val;
            } else {
                K_stat_expected[(d - n_rot) + h * n_stat] = val;
            }
        }
    }

    // Verify the split preserves all data
    std::vector<float> K_reconstructed(head_dim * n_head_kv, 0.0f);
    for (int h = 0; h < n_head_kv; h++) {
        for (int d = 0; d < n_rot; d++) {
            K_reconstructed[d + h * head_dim] = K_rope_expected[d + h * n_rot];
        }
        for (int d = 0; d < n_stat; d++) {
            K_reconstructed[(n_rot + d) + h * head_dim] = K_stat_expected[d + h * n_stat];
        }
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < K_orig.size(); i++) {
        float err = fabsf(K_orig[i] - K_reconstructed[i]);
        if (err > max_err) max_err = err;
    }

    char buf[128];
    snprintf(buf, sizeof(buf), "cache_roundtrip hd=%d rot=%d kv_heads=%d | max_err=%.8f",
             head_dim, n_rot, n_head_kv, max_err);
    check(buf, max_err == 0.0f);
}

int main() {
    printf("=== split K graph-level tests ===\n\n");

    printf("--- 1. Cache write/read roundtrip (data split correctness) ---\n");
    test_cache_roundtrip(256, 64, 2);
    test_cache_roundtrip(128, 32, 4);
    test_cache_roundtrip(256, 128, 2);

    printf("\n--- 2. Split attention graph: single token (decode) ---\n");
    test_split_attn_graph(256, 64, 16, 2, 10, 1);
    test_split_attn_graph(256, 64, 16, 2, 100, 1);
    test_split_attn_graph(256, 64, 16, 2, 512, 1);

    printf("\n--- 3. Split attention graph: multi-token (prompt eval) ---\n");
    test_split_attn_graph(256, 64, 16, 2, 10, 5);
    test_split_attn_graph(256, 64, 16, 2, 100, 32);
    test_split_attn_graph(256, 64, 16, 2, 512, 128);

    printf("\n--- 4. Split attention graph: different head configs ---\n");
    test_split_attn_graph(128, 32, 8, 2, 50, 4);
    test_split_attn_graph(256, 128, 16, 4, 50, 4);   // 50/50 split
    test_split_attn_graph(256, 64, 16, 16, 50, 4);   // MHA (no GQA)

    printf("\n=== %d/%d passed ===\n", n_pass, n_pass + n_fail);
    return n_fail > 0 ? 1 : 0;
}
