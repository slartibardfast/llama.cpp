/*
 * test-flash-attn-lse-merge.cpp
 *
 * Unit test for the online-softmax merge of two ggml_flash_attn_ext_lse
 * outputs. This mirrors what the residual-window read path needs:
 *   Pass A: FA_LSE over keys [0, split)
 *   Pass B: FA_LSE over keys [split, K)
 *   Merge:  combine (VKQ_a, M_a, S_a) with (VKQ_b, M_b, S_b) to recover
 *           the result of a single FA over [0, K).
 *
 * Merge formula (per (head, query)):
 *   M_new  = max(M_a, M_b)
 *   sa     = exp(M_a - M_new)          -- rescale Pass A
 *   sb     = exp(M_b - M_new)          -- rescale Pass B
 *   S_new  = sa*S_a + sb*S_b
 *   VKQ    = sa*VKQ_a + sb*VKQ_b       -- broadcast sa,sb over DV
 *   output = VKQ / S_new
 *
 * Robustness: M is clamped to >= -1e30 before the elementwise max to
 * avoid NaN from (-inf + +inf) when one pass is entirely masked. For a
 * pass with an empty visible set, the kernel emits M = -inf and S = 0;
 * after clamping, sa (or sb) goes to exp(-1e30 - M_new) ≈ 0 and the
 * merged result correctly hands over to the other pass.
 *
 * Verification: compare merged result to a single full-range FA.
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

// Build the merge graph. `lse_a` and `lse_b` must both have shape
// [DV+4, H, N, ne3] — VKQ at [0..DV), M at DV, S at DV+1, pad at
// DV+2..DV+4. Returns a [DV, H, N, ne3] tensor that equals the
// normalised FA output over the union of the two input key ranges.
//
// Implementation leans on existing ggml ops only: add, sub, scale, abs,
// clamp, exp, mul, div. No new kernels required.
static ggml_tensor * build_fa_lse_merge(
        ggml_context * ctx,
        ggml_tensor  * lse_a,
        ggml_tensor  * lse_b) {
    GGML_ASSERT(lse_a->type == GGML_TYPE_F32);
    GGML_ASSERT(lse_b->type == GGML_TYPE_F32);
    GGML_ASSERT(lse_a->ne[0] == lse_b->ne[0]);
    GGML_ASSERT(lse_a->ne[1] == lse_b->ne[1]);
    GGML_ASSERT(lse_a->ne[2] == lse_b->ne[2]);
    GGML_ASSERT(lse_a->ne[3] == lse_b->ne[3]);
    GGML_ASSERT(lse_a->ne[0] >= 5);

    const int64_t DV  = lse_a->ne[0] - 4;
    const int64_t H   = lse_a->ne[1];
    const int64_t N   = lse_a->ne[2];
    const int64_t ne3 = lse_a->ne[3];
    const size_t  es  = ggml_element_size(lse_a);

    auto slice = [&](ggml_tensor * src, int64_t row_offset, int64_t rows) {
        ggml_tensor * v = ggml_view_4d(ctx, src,
                rows, H, N, ne3,
                src->nb[1], src->nb[2], src->nb[3],
                row_offset * es);
        return ggml_cont(ctx, v);
    };

    // VKQ_a,b are [DV, H, N, ne3]; M_a,b and S_a,b are [1, H, N, ne3].
    ggml_tensor * VKQ_a = slice(lse_a, 0,      DV);
    ggml_tensor * VKQ_b = slice(lse_b, 0,      DV);
    ggml_tensor * M_a   = slice(lse_a, DV,     1);
    ggml_tensor * M_b   = slice(lse_b, DV,     1);
    ggml_tensor * S_a   = slice(lse_a, DV + 1, 1);
    ggml_tensor * S_b   = slice(lse_b, DV + 1, 1);

    // Clamp M to >= -1e30 so (-inf) inputs from an empty pass collapse
    // to a finite sentinel before the max. Without this, both the
    // |diff| branch (0.5*(a+b+|a-b|)) and the add-relu branch produce
    // NaN when one side is -inf.
    const float M_FLOOR = -1.0e30f;
    M_a = ggml_clamp(ctx, M_a, M_FLOOR, INFINITY);
    M_b = ggml_clamp(ctx, M_b, M_FLOOR, INFINITY);

    // M_new = 0.5 * (M_a + M_b + |M_a - M_b|)
    ggml_tensor * sum  = ggml_add(ctx, M_a, M_b);
    ggml_tensor * diff = ggml_sub(ctx, M_a, M_b);
    diff               = ggml_abs(ctx, diff);
    ggml_tensor * M_new = ggml_scale(ctx, ggml_add(ctx, sum, diff), 0.5f);

    // sa = exp(M_a - M_new), sb = exp(M_b - M_new)
    ggml_tensor * sa = ggml_exp(ctx, ggml_sub(ctx, M_a, M_new));
    ggml_tensor * sb = ggml_exp(ctx, ggml_sub(ctx, M_b, M_new));

    // S_new = sa*S_a + sb*S_b
    ggml_tensor * S_new = ggml_add(ctx,
            ggml_mul(ctx, sa, S_a),
            ggml_mul(ctx, sb, S_b));

    // VKQ = sa*VKQ_a + sb*VKQ_b (ggml_mul broadcasts sa over DV since
    // sa->ne[0] == 1 and VKQ_a->ne[0] == DV).
    ggml_tensor * VKQ = ggml_add(ctx,
            ggml_mul(ctx, VKQ_a, sa),
            ggml_mul(ctx, VKQ_b, sb));

    // output = VKQ / S_new (same broadcast across DV).
    return ggml_div(ctx, VKQ, S_new);
}

struct test_case {
    const char * name;
    int64_t DK;
    int64_t DV;
    int64_t H;
    int64_t N;
    int64_t K;
    int64_t split;
    float   scale;
    bool    causal;
};

static int run_case(const test_case & tc) {
    fprintf(stdout, "case: %s (DK=%lld DV=%lld H=%lld N=%lld K=%lld split=%lld %s)\n",
            tc.name,
            (long long) tc.DK, (long long) tc.DV, (long long) tc.H,
            (long long) tc.N, (long long) tc.K, (long long) tc.split,
            tc.causal ? "causal" : "unmasked");

    GGML_ASSERT(tc.split >= 1 && tc.split < tc.K);
    const int64_t Ka = tc.split;
    const int64_t Kb = tc.K - tc.split;

    std::mt19937 rng(0xBEEF + (uint32_t) strlen(tc.name));
    std::uniform_real_distribution<float> urand(-1.0f, 1.0f);

    const size_t q_n  = tc.DK * tc.N * tc.H;
    const size_t kF_n = tc.DK * tc.K * tc.H;
    const size_t vF_n = tc.DV * tc.K * tc.H;
    const size_t mF_n = tc.K  * tc.N;

    std::vector<float>       q_host(q_n);
    std::vector<ggml_fp16_t> k_full(kF_n), v_full(vF_n), m_full(mF_n);

    for (auto & x : q_host) x = urand(rng);
    for (auto & x : k_full) x = f32_to_f16(urand(rng));
    for (auto & x : v_full) x = f32_to_f16(urand(rng));

    // Build full causal mask: query n sees keys [0, K - N + n + 1).
    for (int64_t n = 0; n < tc.N; ++n) {
        for (int64_t k = 0; k < tc.K; ++k) {
            const bool visible = tc.causal ? (k < tc.K - tc.N + n + 1) : true;
            m_full[n * tc.K + k] = f32_to_f16(visible ? 0.0f : -INFINITY);
        }
    }

    // Split K/V/mask per head. K layout is [DK, K, H]; index by h*K + k
    // within a per-DK row band. mask layout is [K, N] (shared across H).
    std::vector<ggml_fp16_t> k_a(tc.DK * Ka * tc.H), k_b(tc.DK * Kb * tc.H);
    std::vector<ggml_fp16_t> v_a(tc.DV * Ka * tc.H), v_b(tc.DV * Kb * tc.H);
    std::vector<ggml_fp16_t> m_a(Ka * tc.N),         m_b(Kb * tc.N);

    for (int64_t h = 0; h < tc.H; ++h) {
        for (int64_t k = 0; k < tc.K; ++k) {
            for (int64_t d = 0; d < tc.DK; ++d) {
                const ggml_fp16_t v = k_full[(h * tc.K + k) * tc.DK + d];
                if (k < Ka) {
                    k_a[(h * Ka + k) * tc.DK + d] = v;
                } else {
                    k_b[(h * Kb + (k - Ka)) * tc.DK + d] = v;
                }
            }
            for (int64_t d = 0; d < tc.DV; ++d) {
                const ggml_fp16_t v = v_full[(h * tc.K + k) * tc.DV + d];
                if (k < Ka) {
                    v_a[(h * Ka + k) * tc.DV + d] = v;
                } else {
                    v_b[(h * Kb + (k - Ka)) * tc.DV + d] = v;
                }
            }
        }
    }
    for (int64_t n = 0; n < tc.N; ++n) {
        for (int64_t k = 0; k < tc.K; ++k) {
            if (k < Ka) {
                m_a[n * Ka + k]        = m_full[n * tc.K + k];
            } else {
                m_b[n * Kb + (k - Ka)] = m_full[n * tc.K + k];
            }
        }
    }

    ggml_init_params iparams = { 256 * 1024 * 1024, nullptr, /* no_alloc */ true };
    ggml_context * ctx = ggml_init(iparams);
    assert(ctx);

    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tc.DK, tc.N,  tc.H, 1);
    ggml_tensor * k_ful = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.DK, tc.K,  tc.H, 1);
    ggml_tensor * v_ful = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.DV, tc.K,  tc.H, 1);
    ggml_tensor * m_ful = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.K,  tc.N,  1,    1);

    ggml_tensor * k_aa = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.DK, Ka, tc.H, 1);
    ggml_tensor * v_aa = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.DV, Ka, tc.H, 1);
    ggml_tensor * m_aa = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, Ka,    tc.N, 1,  1);
    ggml_tensor * k_bb = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.DK, Kb, tc.H, 1);
    ggml_tensor * v_bb = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.DV, Kb, tc.H, 1);
    ggml_tensor * m_bb = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, Kb,    tc.N, 1,  1);

    ggml_tensor * out_full = ggml_flash_attn_ext    (ctx, q, k_ful, v_ful, m_ful, tc.scale, 0.0f, 0.0f);
    ggml_tensor * out_a    = ggml_flash_attn_ext_lse(ctx, q, k_aa,  v_aa,  m_aa,  tc.scale, 0.0f, 0.0f);
    ggml_tensor * out_b    = ggml_flash_attn_ext_lse(ctx, q, k_bb,  v_bb,  m_bb,  tc.scale, 0.0f, 0.0f);
    ggml_tensor * merged   = build_fa_lse_merge(ctx, out_a, out_b);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(gf, out_full);
    ggml_build_forward_expand(gf, merged);

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buf);

    ggml_backend_tensor_set(q,     q_host.data(), 0, q_n  * sizeof(float));
    ggml_backend_tensor_set(k_ful, k_full.data(), 0, kF_n * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v_ful, v_full.data(), 0, vF_n * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(m_ful, m_full.data(), 0, mF_n * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(k_aa,  k_a.data(),    0, k_a.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v_aa,  v_a.data(),    0, v_a.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(m_aa,  m_a.data(),    0, m_a.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(k_bb,  k_b.data(),    0, k_b.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v_bb,  v_b.data(),    0, v_b.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(m_bb,  m_b.data(),    0, m_b.size() * sizeof(ggml_fp16_t));

    ggml_backend_cpu_set_n_threads(backend, 4);
    ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stdout, "  FAIL: graph compute status=%d\n", (int) status);
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> full_out  (tc.DV * tc.H * tc.N);
    std::vector<float> merged_out(tc.DV * tc.H * tc.N);
    ggml_backend_tensor_get(out_full, full_out.data(),   0, full_out.size()   * sizeof(float));
    ggml_backend_tensor_get(merged,   merged_out.data(), 0, merged_out.size() * sizeof(float));

    int fails = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    for (int64_t n = 0; n < tc.N; ++n) {
        for (int64_t h = 0; h < tc.H; ++h) {
            for (int64_t d = 0; d < tc.DV; ++d) {
                const size_t idx    = (n * tc.H + h) * tc.DV + d;
                const float  full   = full_out[idx];
                const float  merged = merged_out[idx];
                if (std::isnan(merged) || std::isinf(merged)) {
                    if (fails < 4) {
                        fprintf(stdout, "  FAIL: non-finite merged (n=%lld h=%lld d=%lld) merged=%g full=%g\n",
                                (long long) n, (long long) h, (long long) d, merged, full);
                    }
                    fails++;
                    continue;
                }
                const float diff  = std::abs(merged - full);
                const float denom = std::max(std::abs(full), 1e-6f);
                const float rel   = diff / denom;
                max_abs_diff = std::max(max_abs_diff, diff);
                max_rel_diff = std::max(max_rel_diff, rel);
                if (diff > 1e-3f && rel > 1e-2f) {
                    if (fails < 4) {
                        fprintf(stdout, "  FAIL: (n=%lld h=%lld d=%lld) merged=%g full=%g diff=%g rel=%g\n",
                                (long long) n, (long long) h, (long long) d,
                                merged, full, diff, rel);
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

// Specialised case: Pass A is entirely masked (simulating seq_len <= rw
// in the residual-window read path). The merge must hand over completely
// to Pass B and must not emit NaN from the (-inf, -inf) path.
static int run_empty_pass_a_case() {
    test_case tc = { "empty_pass_a_handover", 64, 64, 2, 1, 16, 8, 1.0f/8.0f, true };
    fprintf(stdout, "case: %s (DK=%lld DV=%lld H=%lld N=%lld K=%lld split=%lld empty-A)\n",
            tc.name,
            (long long) tc.DK, (long long) tc.DV, (long long) tc.H,
            (long long) tc.N, (long long) tc.K, (long long) tc.split);

    const int64_t Ka = tc.split;
    const int64_t Kb = tc.K - tc.split;

    std::mt19937 rng(0xDEAD);
    std::uniform_real_distribution<float> urand(-1.0f, 1.0f);

    std::vector<float>       q_host(tc.DK * tc.N * tc.H);
    std::vector<ggml_fp16_t> k_full(tc.DK * tc.K * tc.H);
    std::vector<ggml_fp16_t> v_full(tc.DV * tc.K * tc.H);
    std::vector<ggml_fp16_t> m_full(tc.K  * tc.N);
    for (auto & x : q_host) x = urand(rng);
    for (auto & x : k_full) x = f32_to_f16(urand(rng));
    for (auto & x : v_full) x = f32_to_f16(urand(rng));

    // Full mask: Pass A range [0, Ka) is all -inf (entire first half
    // masked out). Pass B range [Ka, K) is all visible.
    for (int64_t n = 0; n < tc.N; ++n) {
        for (int64_t k = 0; k < tc.K; ++k) {
            m_full[n * tc.K + k] = f32_to_f16(k < Ka ? -INFINITY : 0.0f);
        }
    }

    // Splits. Pass A inherits the -inf mask → M=-inf, S=0, VKQ=0 from FA_LSE.
    std::vector<ggml_fp16_t> k_a(tc.DK * Ka * tc.H), k_b(tc.DK * Kb * tc.H);
    std::vector<ggml_fp16_t> v_a(tc.DV * Ka * tc.H), v_b(tc.DV * Kb * tc.H);
    std::vector<ggml_fp16_t> m_a(Ka * tc.N),         m_b(Kb * tc.N);
    for (int64_t h = 0; h < tc.H; ++h) {
        for (int64_t k = 0; k < tc.K; ++k) {
            for (int64_t d = 0; d < tc.DK; ++d) {
                const ggml_fp16_t val = k_full[(h * tc.K + k) * tc.DK + d];
                if (k < Ka) k_a[(h * Ka + k) * tc.DK + d]              = val;
                else        k_b[(h * Kb + (k - Ka)) * tc.DK + d]       = val;
            }
            for (int64_t d = 0; d < tc.DV; ++d) {
                const ggml_fp16_t val = v_full[(h * tc.K + k) * tc.DV + d];
                if (k < Ka) v_a[(h * Ka + k) * tc.DV + d]              = val;
                else        v_b[(h * Kb + (k - Ka)) * tc.DV + d]       = val;
            }
        }
    }
    for (int64_t n = 0; n < tc.N; ++n) {
        for (int64_t k = 0; k < tc.K; ++k) {
            if (k < Ka) m_a[n * Ka + k]        = m_full[n * tc.K + k];
            else        m_b[n * Kb + (k - Ka)] = m_full[n * tc.K + k];
        }
    }

    ggml_init_params iparams = { 128 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(iparams);
    assert(ctx);

    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tc.DK, tc.N,  tc.H, 1);
    ggml_tensor * k_ful = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.DK, tc.K,  tc.H, 1);
    ggml_tensor * v_ful = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.DV, tc.K,  tc.H, 1);
    ggml_tensor * m_ful = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.K,  tc.N,  1,    1);
    ggml_tensor * k_aa  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.DK, Ka,    tc.H, 1);
    ggml_tensor * v_aa  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.DV, Ka,    tc.H, 1);
    ggml_tensor * m_aa  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, Ka,    tc.N,  1,    1);
    ggml_tensor * k_bb  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.DK, Kb,    tc.H, 1);
    ggml_tensor * v_bb  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.DV, Kb,    tc.H, 1);
    ggml_tensor * m_bb  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, Kb,    tc.N,  1,    1);

    ggml_tensor * out_full = ggml_flash_attn_ext    (ctx, q, k_ful, v_ful, m_ful, tc.scale, 0.0f, 0.0f);
    ggml_tensor * out_a    = ggml_flash_attn_ext_lse(ctx, q, k_aa,  v_aa,  m_aa,  tc.scale, 0.0f, 0.0f);
    ggml_tensor * out_b    = ggml_flash_attn_ext_lse(ctx, q, k_bb,  v_bb,  m_bb,  tc.scale, 0.0f, 0.0f);
    ggml_tensor * merged   = build_fa_lse_merge(ctx, out_a, out_b);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(gf, out_full);
    ggml_build_forward_expand(gf, merged);

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buf);

    ggml_backend_tensor_set(q,     q_host.data(), 0, q_host.size() * sizeof(float));
    ggml_backend_tensor_set(k_ful, k_full.data(), 0, k_full.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v_ful, v_full.data(), 0, v_full.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(m_ful, m_full.data(), 0, m_full.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(k_aa,  k_a.data(),    0, k_a.size()    * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v_aa,  v_a.data(),    0, v_a.size()    * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(m_aa,  m_a.data(),    0, m_a.size()    * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(k_bb,  k_b.data(),    0, k_b.size()    * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v_bb,  v_b.data(),    0, v_b.size()    * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(m_bb,  m_b.data(),    0, m_b.size()    * sizeof(ggml_fp16_t));

    ggml_backend_cpu_set_n_threads(backend, 4);
    ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stdout, "  FAIL: graph compute status=%d\n", (int) status);
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> full_out  (tc.DV * tc.H * tc.N);
    std::vector<float> merged_out(tc.DV * tc.H * tc.N);
    ggml_backend_tensor_get(out_full, full_out.data(),   0, full_out.size()   * sizeof(float));
    ggml_backend_tensor_get(merged,   merged_out.data(), 0, merged_out.size() * sizeof(float));

    int fails = 0;
    for (size_t i = 0; i < merged_out.size(); ++i) {
        if (std::isnan(merged_out[i]) || std::isinf(merged_out[i])) {
            if (fails < 4) {
                fprintf(stdout, "  FAIL: non-finite merged[%zu]=%g full=%g\n",
                        i, merged_out[i], full_out[i]);
            }
            fails++;
            continue;
        }
        const float diff  = std::abs(merged_out[i] - full_out[i]);
        const float denom = std::max(std::abs(full_out[i]), 1e-6f);
        const float rel   = diff / denom;
        if (diff > 1e-3f && rel > 1e-2f) {
            if (fails < 4) {
                fprintf(stdout, "  FAIL: [%zu] merged=%g full=%g diff=%g rel=%g\n",
                        i, merged_out[i], full_out[i], diff, rel);
            }
            fails++;
        }
    }
    fprintf(stdout, "  empty-pass-A handover: fails=%d\n", fails);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return fails > 0 ? 1 : 0;
}

int main(int argc, char ** argv) {
    (void) argc; (void) argv;
    ggml_backend_load_all();

    const test_case cases[] = {
        { "split_mid_causal",    64,  64, 2, 1,  16,    8, 1.0f/8.0f,   true  },
        { "split_mid_unmasked",  64,  64, 2, 1,  16,    8, 1.0f/8.0f,   false },
        { "split_early",         64,  64, 2, 1,  64,    4, 1.0f/8.0f,   true  },
        { "split_late",          64,  64, 2, 1,  64,   60, 1.0f/8.0f,   true  },
        { "qwen_head_decode",   128, 128, 7, 1, 256,  128, 1.0f/11.31f, true  },
        { "qwen_head_pp4",      128, 128, 7, 4, 256,  128, 1.0f/11.31f, true  },
        { "long_ctx_split",     128, 128, 4, 1, 512,  384, 1.0f/11.31f, true  },
        // PP-scale case matching llama-perplexity workload (batch=512,
        // Qwen3.5 0.8B shape). Exercises the path that shows a
        // multi-thread regression at rw=128 in the residual-window
        // two-pass FA dispatch.
        { "qwen_pp512_rw128",   128, 128,16, 512, 512, 384, 1.0f/11.31f, true  },
    };
    const int n = sizeof(cases) / sizeof(cases[0]);

    int total_fails = 0;
    for (int i = 0; i < n; ++i) {
        total_fails += run_case(cases[i]);
    }
    total_fails += run_empty_pass_a_case();

    fprintf(stdout, "\n=== Summary ===\n  cases: %d\n  failed cases: %d\n",
            n + 1, total_fails);
    return total_fails == 0 ? 0 : 1;
}
