/*
 * test-split-k-cache-ops.cpp — ggml op-level test for split K cache
 *
 * Tests the actual ggml ops used by the split K write/read path:
 *   - ggml_view_3d + ggml_cont (strided split of Kcur)
 *   - ggml_set_rows (scatter-write to cache)
 *   - ggml_view_3d (read from cache)
 *   - ggml_cast (dequant)
 *   - ggml_reshape_4d + ggml_concat
 *
 * Exercises the exact tensor shapes and index patterns from the
 * real KV cache at c=512 and c=2048 to reproduce the OOB crash.
 *
 * Build: cmake --build build-tq --target test-split-k-cache-ops
 * Run:   build-tq/bin/test-split-k-cache-ops
 */

#include "ggml.h"
#include "ggml-cpu.h"

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
 * Test: split K cache write + read roundtrip via ggml ops
 *
 * Simulates what build_attn does:
 * 1. Create Kcur [head_dim, n_head_kv, n_tokens]
 * 2. Split into rope [n_rot, n_head_kv, n_tokens] + static [n_stat, ...]
 * 3. ggml_cont each (strided → contiguous)
 * 4. Flatten to 2D [n_rot_gqa, n_tokens] / [n_stat_gqa, n_tokens]
 * 5. ggml_set_rows into cache tensors using index tensor
 * 6. Read back via ggml_view_3d
 * 7. Cast to f32, reshape to 4D
 * 8. Concat → verify matches original Kcur
 * ========================================================== */
static void test_cache_write_read(
    int head_dim, int n_rot, int n_head_kv,
    int kv_size, int n_tokens,
    const char * quant_name)
{
    const int n_stat = head_dim - n_rot;
    const int n_rot_gqa = n_rot * n_head_kv;
    const int n_stat_gqa = n_stat * n_head_kv;
    const int n_embd_gqa = head_dim * n_head_kv;

    size_t mem_size = 512 * 1024 * 1024;
    struct ggml_init_params params = { mem_size, NULL, false };
    struct ggml_context * ctx = ggml_init(params);

    // === Create source: Kcur [head_dim, n_head_kv, n_tokens] ===
    struct ggml_tensor * Kcur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
        head_dim, n_head_kv, n_tokens);

    uint32_t seed = 0xDEAD0001;
    float * kcur_data = (float *) Kcur->data;
    for (int i = 0; i < head_dim * n_head_kv * n_tokens; i++) {
        kcur_data[i] = det_rand(&seed);
    }

    // === Create cache tensors (f32 for simplicity) ===
    struct ggml_tensor * cache_k_rope = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
        n_rot_gqa, kv_size, 1);
    struct ggml_tensor * cache_k_stat = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
        n_stat_gqa, kv_size, 1);
    memset(cache_k_rope->data, 0, ggml_nbytes(cache_k_rope));
    memset(cache_k_stat->data, 0, ggml_nbytes(cache_k_stat));

    // === Create index tensor (sequential: 0, 1, 2, ..., n_tokens-1) ===
    struct ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    int64_t * idx_data = (int64_t *) k_idxs->data;
    for (int i = 0; i < n_tokens; i++) {
        idx_data[i] = i;  // sequential positions
    }

    // === Split Kcur: rope view [n_rot, n_head_kv, n_tokens] ===
    struct ggml_tensor * k_rope_cur = ggml_view_3d(ctx, Kcur,
        n_rot, n_head_kv, n_tokens,
        Kcur->nb[1], Kcur->nb[2], 0);
    k_rope_cur = ggml_cont(ctx, k_rope_cur);

    // === Split Kcur: static view [n_stat, n_head_kv, n_tokens] ===
    struct ggml_tensor * k_stat_cur = ggml_view_3d(ctx, Kcur,
        n_stat, n_head_kv, n_tokens,
        Kcur->nb[1], Kcur->nb[2],
        n_rot * ggml_element_size(Kcur));
    k_stat_cur = ggml_cont(ctx, k_stat_cur);

    // === Flatten to 2D for set_rows ===
    k_rope_cur = ggml_view_2d(ctx, k_rope_cur,
        n_rot_gqa, n_tokens, k_rope_cur->nb[2], 0);
    k_stat_cur = ggml_view_2d(ctx, k_stat_cur,
        n_stat_gqa, n_tokens, k_stat_cur->nb[2], 0);

    // === Write to cache via set_rows ===
    struct ggml_tensor * cache_rope_2d = ggml_reshape_2d(ctx,
        cache_k_rope, n_rot_gqa, kv_size);
    struct ggml_tensor * cache_stat_2d = ggml_reshape_2d(ctx,
        cache_k_stat, n_stat_gqa, kv_size);

    struct ggml_tensor * write_rope = ggml_set_rows(ctx,
        cache_rope_2d, k_rope_cur, k_idxs);
    struct ggml_tensor * write_stat = ggml_set_rows(ctx,
        cache_stat_2d, k_stat_cur, k_idxs);

    // === Read back: view of cache [n_rot_gqa, n_tokens, 1] ===
    // (In reality n_kv might be > n_tokens due to padding, but
    // for this test n_kv = n_tokens)
    struct ggml_tensor * read_rope = ggml_view_3d(ctx, write_rope,
        n_rot_gqa, n_tokens, 1,
        ggml_row_size(write_rope->type, n_rot_gqa),
        ggml_row_size(write_rope->type, n_rot_gqa * kv_size), 0);

    struct ggml_tensor * read_stat = ggml_view_3d(ctx, write_stat,
        n_stat_gqa, n_tokens, 1,
        ggml_row_size(write_stat->type, n_stat_gqa),
        ggml_row_size(write_stat->type, n_stat_gqa * kv_size), 0);

    // === Reshape to 4D ===
    struct ggml_tensor * rope_4d = ggml_reshape_4d(ctx, read_rope,
        n_rot, n_head_kv, n_tokens, 1);
    struct ggml_tensor * stat_4d = ggml_reshape_4d(ctx, read_stat,
        n_stat, n_head_kv, n_tokens, 1);

    // === Concat along dim 0 ===
    struct ggml_tensor * K_reconstructed = ggml_concat(ctx, rope_4d, stat_4d, 0);
    ggml_set_name(K_reconstructed, "K_reconstructed");

    // === Build and run graph ===
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, K_reconstructed);

    int ok = ggml_graph_compute_with_ctx(ctx, gf, 4);

    if (ok != GGML_STATUS_SUCCESS) {
        char buf[128];
        snprintf(buf, sizeof(buf),
            "cache_ops hd=%d rot=%d kv_heads=%d kv=%d tok=%d",
            head_dim, n_rot, n_head_kv, kv_size, n_tokens);
        check(buf, false);
        printf("    DETAIL: graph compute failed with status %d\n", ok);
        ggml_free(ctx);
        return;
    }

    // === Verify: K_reconstructed should match Kcur ===
    // K_reconstructed is [head_dim, n_head_kv, n_tokens, 1]
    float * recon = (float *) K_reconstructed->data;
    float max_err = 0.0f;
    int worst = -1;
    for (int i = 0; i < head_dim * n_head_kv * n_tokens; i++) {
        float err = fabsf(kcur_data[i] - recon[i]);
        if (err > max_err) { max_err = err; worst = i; }
    }

    char buf[256];
    snprintf(buf, sizeof(buf),
        "cache_ops hd=%d rot=%d kv_heads=%d kv=%d tok=%d | max_err=%.2e @%d",
        head_dim, n_rot, n_head_kv, kv_size, n_tokens, max_err, worst);
    check(buf, max_err < 1e-6f);

    ggml_free(ctx);
}

int main() {
    printf("=== split K cache op-level tests ===\n\n");

    printf("--- 1. Cache write/read: 35B-A3B config (works) ---\n");
    // 35B-A3B: head_dim=256, n_rot=64, n_head_kv=2
    test_cache_write_read(256, 64, 2,  512,   1, "f32");   // single token
    test_cache_write_read(256, 64, 2,  512,  32, "f32");   // small batch
    test_cache_write_read(256, 64, 2,  512, 512, "f32");   // full context

    printf("\n--- 2. Cache write/read: 27B config (crashes at c=2048) ---\n");
    // 27B: head_dim=256, n_rot=64, n_head_kv=4
    test_cache_write_read(256, 64, 4,  512,   1, "f32");   // single token, c=512
    test_cache_write_read(256, 64, 4,  512, 512, "f32");   // full context c=512
    test_cache_write_read(256, 64, 4, 1024,   1, "f32");   // single token, c=1024
    test_cache_write_read(256, 64, 4, 1024,1024, "f32");   // full context c=1024
    test_cache_write_read(256, 64, 4, 2048,   1, "f32");   // single token, c=2048
    test_cache_write_read(256, 64, 4, 2048, 512, "f32");   // partial fill c=2048
    test_cache_write_read(256, 64, 4, 2048,2048, "f32");   // FULL CONTEXT c=2048

    printf("\n--- 3. Stress: large batch sizes ---\n");
    test_cache_write_read(256, 64, 4, 4096,2048, "f32");   // half fill 4K
    test_cache_write_read(256, 64, 4, 4096,4096, "f32");   // full 4K

    printf("\n=== %d/%d passed ===\n", n_pass, n_pass + n_fail);
    return n_fail > 0 ? 1 : 0;
}
