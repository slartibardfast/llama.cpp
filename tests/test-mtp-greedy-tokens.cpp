/*
 * test-mtp-greedy-tokens.cpp — reproduces MTP buffer aliasing bug
 *
 * The MTP head does: argmax(logits) → get_rows(tok_embd, greedy_tokens).
 * When the graph is reserved with n_tokens_reserved > n_tokens_actual,
 * the argmax result buffer can be reused by a subsequent float op,
 * overwriting I32 token IDs with float bit patterns → OOB in get_rows.
 *
 * This test builds a mini graph that mirrors the MTP pattern:
 *   logits = mul_mat(lm_head, hidden_state)
 *   greedy_tokens = argmax(logits)
 *   [intermediate float ops that can alias the buffer]
 *   emb = get_rows(tok_embd, greedy_tokens)
 *
 * Tests both with and without ggml_set_output protection.
 *
 * Build: cmake --build build-tq --target test-mtp-greedy-tokens
 * Run:   build-tq/bin/test-mtp-greedy-tokens
 */

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>

static int n_pass = 0, n_fail = 0;

static void check(const char * name, bool cond) {
    printf("  %-65s %s\n", name, cond ? "PASS" : "FAIL");
    if (cond) n_pass++; else n_fail++;
}

static float det_rand(uint32_t * state) {
    *state = *state * 1103515245u + 12345u;
    return ((float)(*state >> 16) / 32768.0f) - 1.0f;
}

/* ==========================================================
 * Test: argmax → get_rows with buffer pressure
 *
 * Builds a graph that:
 * 1. Computes logits = mul_mat(lm_head, hidden_state)
 *    - hidden_state has n_tokens_reserved rows but only
 *      n_tokens_actual contain valid data (rest = garbage)
 * 2. greedy_tokens = argmax(logits)
 * 3. Several intermediate float ops (to pressure buffer reuse)
 * 4. emb = get_rows(tok_embd, greedy_tokens)
 *
 * Checks that all greedy_tokens values are in [0, n_vocab).
 * ========================================================== */
static void test_mtp_greedy(
    int n_embd, int n_vocab, int n_tokens_actual, int n_tokens_reserved,
    bool protect_output)
{
    size_t mem_size = 256 * 1024 * 1024;
    struct ggml_init_params params = { mem_size, NULL, false };
    struct ggml_context * ctx = ggml_init(params);

    // --- Create weight tensors (simulated model weights) ---
    // lm_head: [n_embd, n_vocab] — projects hidden→logits
    struct ggml_tensor * lm_head = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    // tok_embd: [n_embd, n_vocab] — embedding table
    struct ggml_tensor * tok_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);

    ggml_set_name(lm_head, "lm_head");
    ggml_set_name(tok_embd, "tok_embd");
    ggml_set_input(lm_head);
    ggml_set_input(tok_embd);

    // Fill weights with small random values
    uint32_t seed = 0x4D545001;
    for (int i = 0; i < n_embd * n_vocab; i++) {
        ((float *)lm_head->data)[i] = det_rand(&seed) * 0.01f;
        ((float *)tok_embd->data)[i] = det_rand(&seed) * 0.1f;
    }

    // --- Create hidden_state [n_embd, n_tokens_reserved] ---
    // First n_tokens_actual rows: valid data
    // Remaining rows: garbage (simulates stale graph reservation)
    struct ggml_tensor * hidden_state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens_reserved);
    ggml_set_name(hidden_state, "hidden_state");
    ggml_set_input(hidden_state);

    float * hs_data = (float *)hidden_state->data;
    for (int t = 0; t < n_tokens_reserved; t++) {
        for (int e = 0; e < n_embd; e++) {
            if (t < n_tokens_actual) {
                hs_data[t * n_embd + e] = det_rand(&seed);
            } else {
                // Garbage: large values that could produce OOB argmax
                // if logits are not properly bounded
                hs_data[t * n_embd + e] = (float)(0xDEADBEEF);
            }
        }
    }

    // --- Build graph ---
    // Step 1: logits = mul_mat(lm_head, hidden_state) → [n_vocab, n_tokens_reserved]
    struct ggml_tensor * logits = ggml_mul_mat(ctx, lm_head, hidden_state);
    ggml_set_name(logits, "logits");

    // Step 2: greedy_tokens = argmax(logits) → [n_tokens_reserved] I32
    struct ggml_tensor * greedy_tokens = ggml_argmax(ctx, logits);
    ggml_set_name(greedy_tokens, "greedy_tokens");

    if (protect_output) {
        ggml_set_output(greedy_tokens);
    }

    // Step 3: Intermediate float ops to pressure buffer reuse.
    // Create several temporary f32 tensors sized similarly to greedy_tokens
    // (n_tokens_reserved elements). The allocator may reuse the greedy_tokens
    // buffer for one of these if it's not protected.
    struct ggml_tensor * pressure = logits;
    for (int i = 0; i < 6; i++) {
        pressure = ggml_scale(ctx, pressure, 0.5f);
        pressure = ggml_add(ctx, pressure, logits);
    }
    ggml_set_name(pressure, "pressure");

    // Step 4: emb = get_rows(tok_embd, greedy_tokens) → [n_embd, n_tokens_reserved]
    struct ggml_tensor * emb = ggml_get_rows(ctx, tok_embd, greedy_tokens);
    ggml_set_name(emb, "emb");

    // Step 5: Use both emb and pressure so neither is dead code
    // pressure is [n_vocab, n_tokens_reserved], emb is [n_embd, n_tokens_reserved]
    // Can't add directly — just set both as outputs
    ggml_set_output(emb);
    ggml_set_output(pressure);

    // Build and run
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, emb);
    ggml_build_forward_expand(gf, pressure);

    int status = ggml_graph_compute_with_ctx(ctx, gf, 4);

    // --- Validate ---
    bool compute_ok = (status == GGML_STATUS_SUCCESS);
    if (!compute_ok) {
        char buf[128];
        snprintf(buf, sizeof(buf), "mtp_greedy actual=%d reserved=%d protect=%d compute",
                 n_tokens_actual, n_tokens_reserved, protect_output);
        check(buf, false);
        ggml_free(ctx);
        return;
    }

    // Check all greedy_tokens values are in [0, n_vocab)
    int32_t * gt_data = (int32_t *)greedy_tokens->data;
    int oob_count = 0;
    int32_t worst_val = 0;
    for (int i = 0; i < n_tokens_reserved; i++) {
        if (gt_data[i] < 0 || gt_data[i] >= n_vocab) {
            oob_count++;
            worst_val = gt_data[i];
        }
    }

    // Check valid-row argmax matches reference
    int mismatch_count = 0;
    for (int t = 0; t < n_tokens_actual; t++) {
        float * logit_row = (float *)((char *)logits->data + t * logits->nb[1]);
        int ref_argmax = 0;
        float ref_max = logit_row[0];
        for (int v = 1; v < n_vocab; v++) {
            if (logit_row[v] > ref_max) {
                ref_max = logit_row[v];
                ref_argmax = v;
            }
        }
        if (gt_data[t] != ref_argmax) {
            mismatch_count++;
        }
    }

    char buf[256];
    snprintf(buf, sizeof(buf),
        "mtp_greedy actual=%d reserved=%d protect=%d | oob=%d mismatch=%d",
        n_tokens_actual, n_tokens_reserved, protect_output,
        oob_count, mismatch_count);
    check(buf, oob_count == 0 && mismatch_count == 0);

    if (oob_count > 0) {
        printf("    DETAIL: %d OOB values (worst: %d, vocab: %d)\n",
               oob_count, worst_val, n_vocab);
    }
    if (mismatch_count > 0) {
        printf("    DETAIL: %d valid-row argmax mismatches\n", mismatch_count);
    }

    ggml_free(ctx);
}

int main() {
    printf("=== MTP greedy_tokens buffer aliasing tests ===\n\n");

    // Use small dimensions so tests run fast
    const int n_embd = 64;
    const int n_vocab = 256;

    printf("--- 1. With ggml_set_output protection (the fix) ---\n");
    test_mtp_greedy(n_embd, n_vocab, 512, 512, true);   // no stale rows
    test_mtp_greedy(n_embd, n_vocab, 256, 512, true);   // 50% stale
    test_mtp_greedy(n_embd, n_vocab, 100, 512, true);   // 80% stale
    test_mtp_greedy(n_embd, n_vocab,   1, 512, true);   // decode (1 token)
    test_mtp_greedy(n_embd, n_vocab, 400, 512, true);   // the crash case

    printf("\n--- 2. WITHOUT protection (documents the bug) ---\n");
    test_mtp_greedy(n_embd, n_vocab, 512, 512, false);  // no stale, should pass
    test_mtp_greedy(n_embd, n_vocab, 256, 512, false);  // may fail (buffer reuse)
    test_mtp_greedy(n_embd, n_vocab, 100, 512, false);  // may fail
    test_mtp_greedy(n_embd, n_vocab,   1, 512, false);  // may fail
    test_mtp_greedy(n_embd, n_vocab, 400, 512, false);  // the crash case

    printf("\n=== %d/%d passed ===\n", n_pass, n_pass + n_fail);
    return n_fail > 0 ? 1 : 0;
}
