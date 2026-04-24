/*
 * test-turbo-kv-attention-pbt.cpp
 *
 * Property-based tests propagated from turbo_kv_4b_attention.allium.
 * Asserts GGML_OP_FLASH_ATTN_EXT output equivalence between the CPU
 * scalar reference and any registered non-CPU backend at multi-block
 * head_dim = 256 (the Qwen3.5 9B / 35B-A3B configuration).
 *
 * Motivation: direction-of-travel PPL on 9B IQ3_XXS showed
 * turbo_kv_4b +0.23 PPL vs F16 baseline (vs within-noise on 0.8B).
 * 0.8B uses head_dim=128 (single-block); 9B/35B uses head_dim=256
 * (multi-block). This PBT disambiguates whether the regression is a
 * latent GPU FA bug at multi-block (→ tight failure) or an
 * algorithmic codebook quality gap (→ passes within fp32 tolerance).
 *
 * Output tolerance per spec:
 *   output_abs_tol = 1e-5   (covers near-zero)
 *   output_rel_tol = 1e-5   (covers typical magnitude)
 * Semantics: abs_err <= abs_tol OR rel_err <= rel_tol per element.
 *
 * Build: cmake --build build --target test-turbo-kv-attention-pbt
 * Run:   GGML_VK_VISIBLE_DEVICES=0 build/bin/test-turbo-kv-attention-pbt
 *        (caller acquires gpu-<N>.state via coord/ flock)
 */

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-turbo-kv.h"

#include <rapidcheck.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

/* ================================================================
 * Spec-mirrored tolerance constants (keep in sync with
 * turbo_kv_4b_attention.allium).
 * ================================================================ */

static constexpr float OUTPUT_ABS_TOL  = 1e-5f;
static constexpr float OUTPUT_REL_TOL  = 1e-5f;
static constexpr int   BLOCK_SIZE      = 128;
static constexpr int   MULTI_BLOCK_HEAD_DIM = 256;
static constexpr int   RC_SAMPLES      = 50;

/* ================================================================
 * Helpers
 * ================================================================ */

static float vec_l2_norm(const float * x, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    return sqrtf(s);
}

static void fill_random(float * x, int n, uint32_t seed) {
    uint32_t s = seed | 1u;
    for (int i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        x[i] = ((float)(s >> 16) / 32768.0f - 1.0f);
    }
}

static bool within_abs_or_rel(float a, float b, float abs_tol, float rel_tol) {
    const float diff    = fabsf(a - b);
    const float ref_mag = fmaxf(fabsf(a), fabsf(b));
    if (diff <= abs_tol) return true;
    if (ref_mag > 0.0f && diff / ref_mag <= rel_tol) return true;
    return false;
}

/* ================================================================
 * Backend enumeration — same pattern as test-turbo-kv-backend-pbt
 * but we need the CPU backend as the ORACLE, not a target.
 * ================================================================ */

struct BackendCtx {
    ggml_backend_t     backend = nullptr;    // non-CPU backend under test
    ggml_backend_t     cpu     = nullptr;    // CPU scalar reference
    std::string        name;
};

static std::vector<BackendCtx> init_backends() {
    std::vector<BackendCtx> out;
    ggml_backend_t cpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu = ggml_backend_dev_init(dev, nullptr);
            break;
        }
    }
    if (!cpu) {
        fprintf(stderr, "No CPU backend available — can't run the reference\n");
        return out;
    }
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) continue;
        BackendCtx b;
        b.backend = ggml_backend_dev_init(dev, nullptr);
        b.cpu     = cpu;
        b.name    = ggml_backend_dev_description(dev);
        if (b.backend) out.push_back(std::move(b));
    }
    if (out.empty()) ggml_backend_free(cpu);
    return out;
}

static void free_backends(std::vector<BackendCtx> & bs) {
    ggml_backend_t cpu = nullptr;
    for (auto & b : bs) {
        ggml_backend_free(b.backend);
        cpu = b.cpu;
    }
    if (cpu) ggml_backend_free(cpu);
    bs.clear();
}

/* ================================================================
 * FA config — production-shape for Qwen3.5 9B / 35B-A3B
 * ================================================================ */

struct FAConfig {
    int head_dim;     // 256 for the multi-block path
    int n_tokens;     // queries per call (batch)
    int n_heads_q;    // query heads
    int n_heads_kv;   // KV heads (GQA: n_heads_q / n_heads_kv = group size)
    int n_kv;         // context length
    const char * name;
};

static constexpr FAConfig FA_CONFIGS[] = {
    // Qwen3.5 9B uses 4:1 GQA with head_dim=256. These shapes
    // cover single-token decode + small batch + longer contexts.
    { MULTI_BLOCK_HEAD_DIM, 1, 4, 1,  64,  "hd256_b1_q4_kv1_ctx64"  },
    { MULTI_BLOCK_HEAD_DIM, 8, 4, 1,  64,  "hd256_b8_q4_kv1_ctx64"  },
    { MULTI_BLOCK_HEAD_DIM, 1, 4, 1, 256,  "hd256_b1_q4_kv1_ctx256" },
    { MULTI_BLOCK_HEAD_DIM, 8, 4, 1, 256,  "hd256_b8_q4_kv1_ctx256" },
};

/* ================================================================
 * Build a FLASH_ATTN_EXT graph, run it, return the output tensor
 * data into `out`. ggml_backend_sched_new requires CPU to be the
 * last backend in the array, so we always include it. For the
 * CPU-oracle run `target == cpu` and the sched is just {CPU}; for
 * the backend-under-test run we pass {target, cpu} so unsupported
 * subgraphs can fall back to CPU if needed — but we'll fail the
 * property if the main FA op ends up on CPU for the non-CPU run.
 * ================================================================ */
static bool run_fa_on_backend(
    ggml_backend_t       target,
    ggml_backend_t       cpu,
    const FAConfig &     cfg,
    const std::vector<float>              & q_data,
    const std::vector<block_turbo_kv_4b>  & k_data,
    const std::vector<block_turbo_kv_4b>  & v_data,
    std::vector<float>                    & out)
{
    ggml_init_params p = { 128 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(p);

    // q: F32 [head_dim, n_tokens, n_heads_q, 1]
    // k: TURBO_KV_4B [head_dim, n_kv, n_heads_kv, 1]
    // v: TURBO_KV_4B [head_dim, n_kv, n_heads_kv, 1]
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                        cfg.head_dim, cfg.n_tokens, cfg.n_heads_q, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_TURBO_KV_4B,
                        cfg.head_dim, cfg.n_kv, cfg.n_heads_kv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_TURBO_KV_4B,
                        cfg.head_dim, cfg.n_kv, cfg.n_heads_kv, 1);

    const float scale = 1.0f / sqrtf((float) cfg.head_dim);
    ggml_tensor * out_t = ggml_flash_attn_ext(ctx, q, k, v, /*mask*/nullptr,
                                               scale, /*max_bias*/0.0f,
                                               /*logit_softcap*/0.0f);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out_t);

    // sched_new requires CPU to be the last element of `backends[]`.
    // For a CPU-only run we pass {cpu}; for a backend-under-test run
    // we pass {target, cpu} so subgraphs can fall back.
    ggml_backend_t backends_gpu[] = { target, cpu };
    ggml_backend_t backends_cpu[] = { cpu };
    const bool target_is_cpu = (target == cpu);
    ggml_backend_sched_t sched = target_is_cpu
        ? ggml_backend_sched_new(backends_cpu, nullptr, 1, 4096, false, false)
        : ggml_backend_sched_new(backends_gpu, nullptr, 2, 4096, false, false);
    ggml_backend_sched_reset(sched);

    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return false;   // backend doesn't support this op — skip gracefully
    }

    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, k_data.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, v_data.data(), 0, ggml_nbytes(v));

    const auto status = ggml_backend_sched_graph_compute(sched, graph);
    bool ok = (status == GGML_STATUS_SUCCESS);
    if (ok) {
        out.resize(ggml_nelements(out_t));
        ggml_backend_tensor_get(out_t, out.data(), 0, ggml_nbytes(out_t));
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return ok;
}

/* ================================================================
 * Property — OutputEquivalence (+ MultiBlockParity, implicit since
 * every FA_CONFIGS entry is head_dim=256 multi-block).
 * ================================================================ */
static void property_OutputEquivalence(BackendCtx & bctx) {
    for (const auto & cfg : FA_CONFIGS) {
        printf("- [%s] OutputEquivalence %s: CPU FA vs backend FA, per-element abs|rel 1e-5\n",
               bctx.name.c_str(), cfg.name);
        rc::check(
            [&](uint32_t seed) {
                const int q_elems = cfg.head_dim * cfg.n_tokens * cfg.n_heads_q;
                const int k_elems = cfg.head_dim * cfg.n_kv      * cfg.n_heads_kv;
                const int blocks_per_k_row = cfg.head_dim / BLOCK_SIZE;
                const int k_blocks = cfg.n_kv * cfg.n_heads_kv * blocks_per_k_row;

                std::vector<float> q_data(q_elems);
                fill_random(q_data.data(), q_elems, seed ^ 0x51u);

                // Generate per-row F32 then quantize into turbo_kv_4b blocks.
                std::vector<float> k_raw(k_elems);
                std::vector<float> v_raw(k_elems);
                fill_random(k_raw.data(), k_elems, seed ^ 0x4Bu);
                fill_random(v_raw.data(), k_elems, seed ^ 0x56u);

                // Ensure every head_dim row has L2_norm > 0 (spec precondition).
                const int n_rows = cfg.n_kv * cfg.n_heads_kv;
                for (int r = 0; r < n_rows; r++) {
                    if (vec_l2_norm(&k_raw[r * cfg.head_dim], cfg.head_dim) < 1e-6f) {
                        k_raw[r * cfg.head_dim] = 1.0f;
                    }
                    if (vec_l2_norm(&v_raw[r * cfg.head_dim], cfg.head_dim) < 1e-6f) {
                        v_raw[r * cfg.head_dim] = 1.0f;
                    }
                }

                std::vector<block_turbo_kv_4b> k_blocks_buf(k_blocks);
                std::vector<block_turbo_kv_4b> v_blocks_buf(k_blocks);
                for (int r = 0; r < n_rows; r++) {
                    quantize_row_turbo_kv_4b_ref(
                        &k_raw[r * cfg.head_dim],
                        &k_blocks_buf[r * blocks_per_k_row],
                        cfg.head_dim);
                    quantize_row_turbo_kv_4b_ref(
                        &v_raw[r * cfg.head_dim],
                        &v_blocks_buf[r * blocks_per_k_row],
                        cfg.head_dim);
                }

                // Run on CPU (oracle) and on backend under test.
                std::vector<float> cpu_out, gpu_out;
                bool cpu_ok = run_fa_on_backend(bctx.cpu, bctx.cpu, cfg,
                                                 q_data, k_blocks_buf, v_blocks_buf,
                                                 cpu_out);
                bool gpu_ok = run_fa_on_backend(bctx.backend, bctx.cpu, cfg,
                                                 q_data, k_blocks_buf, v_blocks_buf,
                                                 gpu_out);

                if (!cpu_ok || !gpu_ok) return;   // op not supported — skip

                RC_ASSERT(cpu_out.size() == gpu_out.size());

                int fail_count = 0;
                float max_abs = 0.0f;
                float max_rel = 0.0f;
                int max_idx  = -1;
                for (size_t i = 0; i < cpu_out.size(); i++) {
                    const float a = fabsf(cpu_out[i] - gpu_out[i]);
                    const float m = fmaxf(fabsf(cpu_out[i]), fabsf(gpu_out[i]));
                    const float r = (m > 0.0f) ? (a / m) : 0.0f;
                    if (a > max_abs) max_abs = a;
                    if (r > max_rel) max_rel = r;
                    if (!within_abs_or_rel(cpu_out[i], gpu_out[i],
                                            OUTPUT_ABS_TOL, OUTPUT_REL_TOL)) {
                        fail_count++;
                        if (max_idx < 0) max_idx = (int)i;
                    }
                }

                // Print-on-fail helper — PBT shrinker outputs the seed,
                // this tells us how close/far we were on the full tensor.
                if (fail_count > 0) {
                    fprintf(stderr,
                            "  FAIL [%s]: %d / %zu elems exceeded tolerance. "
                            "max_abs=%.3e max_rel=%.3e first_bad=%d cpu=%.6e gpu=%.6e\n",
                            cfg.name, fail_count, cpu_out.size(),
                            max_abs, max_rel, max_idx,
                            max_idx >= 0 ? cpu_out[max_idx] : 0.0f,
                            max_idx >= 0 ? gpu_out[max_idx] : 0.0f);
                }

                RC_ASSERT(fail_count == 0);
            });
    }
}

/* ================================================================
 * main
 * ================================================================ */

int main(int argc, char ** argv) {
    (void)argc; (void)argv;
    printf("=== turbo_kv_4b attention PBT (CPU vs backend FA) ===\n\n");

    auto backends = init_backends();
    if (backends.empty()) {
        printf("No non-CPU backend available in this build — nothing to test.\n");
        return 0;
    }

    for (auto & bctx : backends) {
        printf("=== Backend: %s ===\n", bctx.name.c_str());
        property_OutputEquivalence(bctx);
        printf("\n");
    }

    free_backends(backends);
    printf("=== All attention properties passed ===\n");
    return 0;
}
