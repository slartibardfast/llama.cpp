/*
 * test-flash-attn-lse.cpp
 *
 * Unit test for ggml_flash_attn_ext_lse — the FA variant that emits
 * unscaled (VKQ, M, S) per query head instead of the normalised output.
 *
 * Contract:
 *   FA_LSE(Q, K, V, mask) -> [DV+2, H, N]
 *     rows [0..DV) per (h, n)   : VKQ_unscaled = sum_k exp(q·k - M) * V_k
 *     row    DV per (h, n)      : M = max_k (q·k)
 *     row    DV+1 per (h, n)    : S = sum_k exp(q·k - M)
 *
 *   FA(Q, K, V, mask) -> [DV, H, N]
 *     rows [0..DV) per (h, n)   : VKQ_unscaled / S
 *
 * Verification: manually divide FA_LSE's VKQ rows by S; compare to FA.
 * Values should agree to within fp32 rounding for the accumulator path.
 *
 * Inputs: small random fp16 K/V, fp32 Q, random causal-style mask.
 * No model load. Pure ggml.
 */

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static ggml_fp16_t f32_to_f16(float x) { return ggml_fp32_to_fp16(x); }
static float       f16_to_f32(ggml_fp16_t x) { return ggml_fp16_to_fp32(x); }

struct test_case {
    const char * name;
    int64_t DK;     // K head dim (also Q head dim)
    int64_t DV;     // V head dim
    int64_t H;      // attention heads
    int64_t N;      // query tokens (typically 1 for decode)
    int64_t K;      // key tokens in cache
    float   scale;
    bool    with_mask;
};

static int run_case(const test_case & tc) {
    fprintf(stdout, "case: %s (DK=%lld DV=%lld H=%lld N=%lld K=%lld %s)\n",
            tc.name,
            (long long) tc.DK, (long long) tc.DV, (long long) tc.H,
            (long long) tc.N, (long long) tc.K,
            tc.with_mask ? "masked" : "unmasked");

    // seeded RNG — deterministic per-case values
    std::mt19937 rng(0xC0FFEE + (uint32_t) strlen(tc.name));
    std::uniform_real_distribution<float> urand(-1.0f, 1.0f);

    // Build input buffers. Post-permute shapes for ggml_flash_attn_ext
    // (see test-backend-ops.cpp::test_flash_attn_ext):
    //   q: [DK, N, H, 1]   (fp32) — ne[1]=n_tokens, ne[2]=n_heads
    //   k: [DK, K, H, 1]   (fp16) — ne[1]=n_kv,     ne[2]=n_heads
    //   v: [DV, K, H, 1]   (fp16) — same as k
    //   mask: [K, N, 1, 1] (fp16) — optional
    const size_t q_n_elems = tc.DK * tc.N * tc.H;
    const size_t k_n_elems = tc.DK * tc.K * tc.H;
    const size_t v_n_elems = tc.DV * tc.K * tc.H;
    const size_t m_n_elems = tc.K  * tc.N;

    std::vector<float>       q_host(q_n_elems);
    std::vector<ggml_fp16_t> k_host(k_n_elems);
    std::vector<ggml_fp16_t> v_host(v_n_elems);
    std::vector<ggml_fp16_t> m_host(m_n_elems);

    for (auto & x : q_host) x = urand(rng);
    for (auto & x : k_host) x = f32_to_f16(urand(rng));
    for (auto & x : v_host) x = f32_to_f16(urand(rng));
    if (tc.with_mask) {
        // causal-style mask: for each query n, keys [0..n+K-N] visible, rest masked
        for (int64_t n = 0; n < tc.N; ++n) {
            for (int64_t k = 0; k < tc.K; ++k) {
                const float val = (k < tc.K - tc.N + n + 1) ? 0.0f : -INFINITY;
                m_host[n * tc.K + k] = f32_to_f16(val);
            }
        }
    }

    // ggml context setup. no_alloc=true because the CPU backend allocates
    // the tensor buffer via ggml_backend_alloc_ctx_tensors.
    ggml_init_params iparams = { 128 * 1024 * 1024, nullptr, /* no_alloc */ true };
    ggml_context * ctx = ggml_init(iparams);
    assert(ctx);

    // Create input tensors. Post-permute convention: ne[1]=n_tokens, ne[2]=n_heads.
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,  tc.DK, tc.N,  tc.H, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16,  tc.DK, tc.K,  tc.H, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16,  tc.DV, tc.K,  tc.H, 1);
    ggml_tensor * m = tc.with_mask
        ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16,  tc.K,  tc.N,  1,    1)
        : nullptr;

    // Build two FA nodes sharing the same inputs.
    ggml_tensor * out_fa  = ggml_flash_attn_ext    (ctx, q, k, v, m, tc.scale, 0.0f, 0.0f);
    ggml_tensor * out_lse = ggml_flash_attn_ext_lse(ctx, q, k, v, m, tc.scale, 0.0f, 0.0f);

    // Check expected shapes.
    if (out_fa->ne[0] != tc.DV) {
        fprintf(stdout, "  FAIL: FA shape[0]=%lld expected %lld\n",
                (long long) out_fa->ne[0], (long long) tc.DV);
        ggml_free(ctx);
        return 1;
    }
    if (out_lse->ne[0] != tc.DV + 2) {
        fprintf(stdout, "  FAIL: FA_LSE shape[0]=%lld expected %lld\n",
                (long long) out_lse->ne[0], (long long) tc.DV + 2);
        ggml_free(ctx);
        return 1;
    }
    if (!ggml_flash_attn_ext_is_lse(out_lse) || ggml_flash_attn_ext_is_lse(out_fa)) {
        fprintf(stdout, "  FAIL: is_lse flag mismatch (fa=%d lse=%d)\n",
                (int) ggml_flash_attn_ext_is_lse(out_fa),
                (int) ggml_flash_attn_ext_is_lse(out_lse));
        ggml_free(ctx);
        return 1;
    }

    // Compute both via a fresh cgraph on the CPU backend.
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_fa);
    ggml_build_forward_expand(gf, out_lse);

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend);
    // Allocate all tensors on the CPU backend (ctx was created with
    // no_alloc=true so tensors have no backing storage until now).
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buf);
    ggml_backend_tensor_set(q, q_host.data(), 0, q_n_elems * sizeof(float));
    ggml_backend_tensor_set(k, k_host.data(), 0, k_n_elems * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v, v_host.data(), 0, v_n_elems * sizeof(ggml_fp16_t));
    if (m) ggml_backend_tensor_set(m, m_host.data(), 0, m_n_elems * sizeof(ggml_fp16_t));

    ggml_backend_cpu_set_n_threads(backend, 4);
    ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stdout, "  FAIL: graph compute status=%d\n", (int) status);
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 1;
    }

    // Pull both outputs to host.
    std::vector<float> fa_out (tc.DV      * tc.H * tc.N);
    std::vector<float> lse_out((tc.DV + 2) * tc.H * tc.N);
    ggml_backend_tensor_get(out_fa,  fa_out.data(),  0, fa_out.size()  * sizeof(float));
    ggml_backend_tensor_get(out_lse, lse_out.data(), 0, lse_out.size() * sizeof(float));

    // Output layout (from FA permute): index = (n * H + h) * row_stride + d.
    // row_stride for FA is DV; for FA_LSE is DV+2.
    int fails = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;

    for (int64_t n = 0; n < tc.N; ++n) {
        for (int64_t h = 0; h < tc.H; ++h) {
            const float * fa_row  = &fa_out [(n * tc.H + h) * tc.DV];
            const float * lse_row = &lse_out[(n * tc.H + h) * (tc.DV + 2)];
            const float   M       = lse_row[tc.DV];
            const float   S       = lse_row[tc.DV + 1];

            // S must be strictly positive for a non-empty visible key set.
            if (S <= 0.0f) {
                if (M != -INFINITY) {
                    fprintf(stdout, "  FAIL: (n=%lld h=%lld) S=%g with non-(-inf) M=%g\n",
                            (long long) n, (long long) h, S, M);
                    fails++;
                }
                continue;
            }

            // manual normalise: VKQ / S should equal FA's output
            for (int64_t d = 0; d < tc.DV; ++d) {
                const float manual = lse_row[d] / S;
                const float fused  = fa_row[d];
                const float diff   = std::abs(manual - fused);
                const float denom  = std::max(std::abs(fused), 1e-6f);
                const float rel    = diff / denom;
                max_abs_diff = std::max(max_abs_diff, diff);
                max_rel_diff = std::max(max_rel_diff, rel);
                // Both paths compute the same math but may accumulate in
                // different orders — FA's split-KV chunked path (nek1>=512
                // at decode) reduces across thread-local partials, while
                // the LSE path forces single-chunk for simplicity. The
                // resulting fp32 non-associativity gives ~1e-4 abs diff
                // on long contexts. Pass if either abs or rel is tight.
                if (diff > 1e-3f && rel > 1e-2f) {
                    if (fails < 4) {
                        fprintf(stdout, "  FAIL: (n=%lld h=%lld d=%lld) manual=%g fa=%g diff=%g rel=%g (M=%g S=%g)\n",
                                (long long) n, (long long) h, (long long) d,
                                manual, fused, diff, rel, M, S);
                    }
                    fails++;
                }
            }
        }
    }

    fprintf(stdout, "  max_abs_diff=%.3e max_rel_diff=%.3e fails=%d\n",
            max_abs_diff, max_rel_diff, fails);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return fails > 0 ? 1 : 0;
}

int main(int argc, char ** argv) {
    (void) argc; (void) argv;
    ggml_backend_load_all();

    const test_case cases[] = {
        { "decode_small_unmasked", 64,  64,  2, 1,   8,  1.0f/8.0f,  false },
        { "decode_small_masked",   64,  64,  2, 1,   8,  1.0f/8.0f,  true  },
        { "decode_qwen_head",     128, 128,  7, 1, 256, 1.0f/11.31f, true  },
        { "pp_batch4",             64,  64,  2, 4,  16,  1.0f/8.0f,  true  },
        { "decode_long_ctx",      128, 128,  4, 1, 512, 1.0f/11.31f, true  },
    };
    const int n_cases = sizeof(cases) / sizeof(cases[0]);

    int total_fails = 0;
    for (int i = 0; i < n_cases; ++i) {
        total_fails += run_case(cases[i]);
    }

    fprintf(stdout, "\n=== Summary ===\n  cases: %d\n  failed cases: %d\n",
            n_cases, total_fails);
    return total_fails == 0 ? 0 : 1;
}
