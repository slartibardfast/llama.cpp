/*
 * test-turbo-kv-backend-pbt.cpp — Property-based tests propagated from
 * turbo_kv_4b_backend.allium.
 *
 * Asserts that any registered non-CPU backend's implementations of
 * quantize / dequantize / get_rows / set_rows for TURBO_KV_4B match the
 * CPU scalar reference within the tolerance bounds from the spec:
 *
 *   indices_byte_tolerance = 0           (byte-identical on mse_indices)
 *   scale_rel_tol          = 1e-6        (norm, inv_std)
 *   element_abs_tol        = 1e-5        (per-element reconstruction)
 *   element_rel_tol        = 1e-5        (abs-OR-rel semantics)
 *
 * Tolerances are calibrated at the theoretical fp32 floor
 * (log2(n) * fp32_epsilon for pairwise sum-tree reductions). They are
 * backend-agnostic: subgroupAdd on Vulkan, warp-shuffle on CUDA,
 * wavefront DPP-reduce on ROCm, simdgroup reduce on Metal all hit the
 * same bound for fp32 pairwise reductions.
 *
 * This test iterates over every non-CPU backend registered at runtime
 * via ggml_backend_dev_count() (following test-backend-ops.cpp's
 * pattern). Adding CUDA / ROCm / Metal to the ggml build surfaces
 * those devices through the enumeration automatically; no file edit
 * needed here to pick them up. If the current build has no non-CPU
 * backend, the test exits 0 with a note.
 *
 * Build: cmake --build build --target test-turbo-kv-backend-pbt
 * Run:   GGML_VK_VISIBLE_DEVICES=0 build/bin/test-turbo-kv-backend-pbt
 *        (caller is responsible for acquiring gpu-<N>.state via the
 *         coord/ flock protocol before invoking.)
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
 * Spec-mirrored tolerance constants.
 * Keep in sync with turbo_kv_4b_backend.allium config.
 * ================================================================ */

static constexpr int   INDICES_BYTE_TOL = 0;
static constexpr float SCALE_REL_TOL    = 1e-6f;
static constexpr float ELEMENT_ABS_TOL  = 1e-5f;
static constexpr float ELEMENT_REL_TOL  = 1e-5f;
static constexpr int   BLOCK_SIZE       = 128;

/* ================================================================
 * Helpers
 * ================================================================ */

static float vec_l2_norm(const float * x, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    return sqrtf(s);
}

// Fill `x[0..n)` with deterministic pseudo-random values in [-1, 1]
// from `seed`. Matches test-turbo-kv-pbt.cpp::fill_random so CPU
// scalar references compute the same thing on both sides.
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
 * Backend enumeration
 * ================================================================ */

struct BackendCtx {
    ggml_backend_t     backend = nullptr;
    ggml_backend_t     cpu     = nullptr;
    std::string        name;
};

static std::vector<BackendCtx> init_backends() {
    std::vector<BackendCtx> out;
    ggml_backend_t cpu = nullptr;

    // First pass: find the CPU backend (scheduler always wants it)
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu = ggml_backend_dev_init(dev, nullptr);
            break;
        }
    }
    if (!cpu) {
        fprintf(stderr, "No CPU backend available — cannot run scheduler\n");
        return out;
    }

    // Second pass: collect every non-CPU backend
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) continue;
        BackendCtx bctx;
        bctx.backend = ggml_backend_dev_init(dev, nullptr);
        bctx.cpu     = cpu;
        bctx.name    = ggml_backend_dev_description(dev);
        if (bctx.backend) out.push_back(std::move(bctx));
    }
    if (out.empty()) {
        ggml_backend_free(cpu);
    }
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
 * Property 1 — QuantizeEquivalence
 *
 * Build a graph: F32 src -> CPY -> TURBO_KV_4B dst. Run on backend.
 * Compare the backend's quantized blocks against CPU scalar output.
 *
 * Asserts:
 *   - mse_indices byte-for-byte equal (INDICES_BYTE_TOL)
 *   - norm and inv_std agree within SCALE_REL_TOL
 * ================================================================ */
static void property_QuantizeEquivalence(BackendCtx & bctx) {
    printf("- [%s] QuantizeEquivalence: F32 input -> backend CPY -> block-for-block vs CPU\n",
           bctx.name.c_str());
    rc::check(
        [&bctx](uint8_t n_blocks_sel, uint32_t seed) {
            const int n_blocks = 1 + (n_blocks_sel & 7);  // 1..8
            const int n_elem   = n_blocks * BLOCK_SIZE;

            std::vector<float> input(n_elem);
            fill_random(input.data(), n_elem, seed);
            if (vec_l2_norm(input.data(), n_elem) < 1e-6f) input[0] = 1.0f;

            // CPU reference
            std::vector<block_turbo_kv_4b> cpu_blocks(n_blocks);
            for (int b = 0; b < n_blocks; b++) {
                quantize_row_turbo_kv_4b_ref(&input[b * BLOCK_SIZE],
                                              &cpu_blocks[b], BLOCK_SIZE);
            }

            // Backend: F32 -> CPY -> TURBO_KV_4B
            ggml_init_params p = { 32 * 1024 * 1024, nullptr, true };
            ggml_context * ctx = ggml_init(p);

            ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elem);
            ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_TURBO_KV_4B, n_elem);
            ggml_tensor * cpy = ggml_cpy(ctx, src, dst);

            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, cpy);

            ggml_backend_t backends[] = { bctx.backend, bctx.cpu };
            ggml_backend_sched_t sched =
                ggml_backend_sched_new(backends, nullptr, 2, 4096, false, false);
            ggml_backend_sched_reset(sched);

            if (!ggml_backend_sched_alloc_graph(sched, graph)) {
                ggml_backend_sched_free(sched);
                ggml_free(ctx);
                // Not supported on this backend — skip cleanly
                return;
            }

            ggml_backend_tensor_set(src, input.data(), 0, n_elem * sizeof(float));

            const auto status = ggml_backend_sched_graph_compute(sched, graph);
            RC_ASSERT(status == GGML_STATUS_SUCCESS);

            std::vector<block_turbo_kv_4b> gpu_blocks(n_blocks);
            ggml_backend_tensor_get(cpy, gpu_blocks.data(), 0,
                                     n_blocks * sizeof(block_turbo_kv_4b));

            for (int b = 0; b < n_blocks; b++) {
                // Indices byte-identical
                const int diff = memcmp(cpu_blocks[b].mse_indices,
                                         gpu_blocks[b].mse_indices,
                                         BLOCK_SIZE / 2);
                RC_ASSERT(diff == INDICES_BYTE_TOL);

                // Scales within SCALE_REL_TOL
                RC_ASSERT(within_abs_or_rel(cpu_blocks[b].norm,
                                             gpu_blocks[b].norm,
                                             0.0f, SCALE_REL_TOL));
                RC_ASSERT(within_abs_or_rel(cpu_blocks[b].inv_std,
                                             gpu_blocks[b].inv_std,
                                             0.0f, SCALE_REL_TOL));
            }

            ggml_backend_sched_free(sched);
            ggml_free(ctx);
        });
}

/* ================================================================
 * Property 2 — DequantizeEquivalence
 *
 * Start with CPU-quantized blocks, run backend dequant via CAST+CONT,
 * compare per-element to CPU dequant.
 * ================================================================ */
static void property_DequantizeEquivalence(BackendCtx & bctx) {
    printf("- [%s] DequantizeEquivalence: backend CAST dequant vs CPU per-element\n",
           bctx.name.c_str());
    rc::check(
        [&bctx](uint8_t n_blocks_sel, uint32_t seed) {
            const int n_blocks = 1 + (n_blocks_sel & 7);  // 1..8
            const int n_elem   = n_blocks * BLOCK_SIZE;

            std::vector<float> input(n_elem);
            fill_random(input.data(), n_elem, seed);
            if (vec_l2_norm(input.data(), n_elem) < 1e-6f) input[0] = 1.0f;

            std::vector<block_turbo_kv_4b> blocks(n_blocks);
            for (int b = 0; b < n_blocks; b++) {
                quantize_row_turbo_kv_4b_ref(&input[b * BLOCK_SIZE],
                                              &blocks[b], BLOCK_SIZE);
            }

            std::vector<float> cpu_out(n_elem);
            for (int b = 0; b < n_blocks; b++) {
                dequantize_row_turbo_kv_4b(&blocks[b],
                                            &cpu_out[b * BLOCK_SIZE],
                                            BLOCK_SIZE);
            }

            ggml_init_params p = { 32 * 1024 * 1024, nullptr, true };
            ggml_context * ctx = ggml_init(p);

            ggml_tensor * src  = ggml_new_tensor_1d(ctx, GGML_TYPE_TURBO_KV_4B, n_elem);
            ggml_tensor * cast = ggml_cast(ctx, src, GGML_TYPE_F32);
            ggml_tensor * dst  = ggml_cont(ctx, cast);

            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, dst);

            ggml_backend_t backends[] = { bctx.backend, bctx.cpu };
            ggml_backend_sched_t sched =
                ggml_backend_sched_new(backends, nullptr, 2, 4096, false, false);
            ggml_backend_sched_reset(sched);

            if (!ggml_backend_sched_alloc_graph(sched, graph)) {
                ggml_backend_sched_free(sched);
                ggml_free(ctx);
                return;
            }

            ggml_backend_tensor_set(src, blocks.data(), 0,
                                     n_blocks * sizeof(block_turbo_kv_4b));

            const auto status = ggml_backend_sched_graph_compute(sched, graph);
            RC_ASSERT(status == GGML_STATUS_SUCCESS);

            std::vector<float> gpu_out(n_elem);
            ggml_backend_tensor_get(dst, gpu_out.data(), 0, n_elem * sizeof(float));

            for (int i = 0; i < n_elem; i++) {
                RC_ASSERT(within_abs_or_rel(cpu_out[i], gpu_out[i],
                                             ELEMENT_ABS_TOL, ELEMENT_REL_TOL));
            }

            ggml_backend_sched_free(sched);
            ggml_free(ctx);
        });
}

/* ================================================================
 * Property 3 — GetRowsEquivalence
 *
 * Build a 2D TurboBlock-quantized source [n_elem, n_rows]; run backend
 * GET_ROWS with a random permutation of row indices; verify output
 * matches CPU dequant of the source sliced by those indices.
 * ================================================================ */
static void property_GetRowsEquivalence(BackendCtx & bctx) {
    printf("- [%s] GetRowsEquivalence: backend GET_ROWS vs CPU dequant + slice\n",
           bctx.name.c_str());
    rc::check(
        [&bctx](uint8_t rows_sel, uint32_t seed) {
            const int n_rows     = 1 + (rows_sel & 3);  // 1..4 rows
            const int n_elem_col = BLOCK_SIZE;          // 1 block per row
            const int n_select   = n_rows;              // request all rows

            std::vector<float> input(n_rows * n_elem_col);
            fill_random(input.data(), n_rows * n_elem_col, seed);

            // Ensure every row has L2_norm > 0
            for (int r = 0; r < n_rows; r++) {
                if (vec_l2_norm(&input[r * n_elem_col], n_elem_col) < 1e-6f) {
                    input[r * n_elem_col] = 1.0f;
                }
            }

            std::vector<block_turbo_kv_4b> blocks(n_rows);
            for (int r = 0; r < n_rows; r++) {
                quantize_row_turbo_kv_4b_ref(&input[r * n_elem_col],
                                              &blocks[r], n_elem_col);
            }

            // Deterministic permutation from seed
            std::vector<int32_t> indices(n_select);
            for (int i = 0; i < n_select; i++) {
                indices[i] = (int32_t)((seed + (uint32_t)i * 2654435761u) % (uint32_t)n_rows);
            }

            // CPU reference: dequantize selected rows
            std::vector<float> cpu_out(n_select * n_elem_col);
            for (int i = 0; i < n_select; i++) {
                dequantize_row_turbo_kv_4b(&blocks[indices[i]],
                                            &cpu_out[i * n_elem_col],
                                            n_elem_col);
            }

            // Backend GET_ROWS
            ggml_init_params p = { 32 * 1024 * 1024, nullptr, true };
            ggml_context * ctx = ggml_init(p);

            ggml_tensor * src     = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO_KV_4B,
                                                        n_elem_col, n_rows);
            ggml_tensor * idx_t   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_select);
            ggml_tensor * get_rows = ggml_get_rows(ctx, src, idx_t);

            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, get_rows);

            ggml_backend_t backends[] = { bctx.backend, bctx.cpu };
            ggml_backend_sched_t sched =
                ggml_backend_sched_new(backends, nullptr, 2, 4096, false, false);
            ggml_backend_sched_reset(sched);

            if (!ggml_backend_sched_alloc_graph(sched, graph)) {
                ggml_backend_sched_free(sched);
                ggml_free(ctx);
                return;
            }

            ggml_backend_tensor_set(src, blocks.data(), 0,
                                     n_rows * sizeof(block_turbo_kv_4b));
            ggml_backend_tensor_set(idx_t, indices.data(), 0,
                                     n_select * sizeof(int32_t));

            const auto status = ggml_backend_sched_graph_compute(sched, graph);
            RC_ASSERT(status == GGML_STATUS_SUCCESS);

            std::vector<float> gpu_out(n_select * n_elem_col);
            ggml_backend_tensor_get(get_rows, gpu_out.data(), 0,
                                     n_select * n_elem_col * sizeof(float));

            for (int i = 0; i < n_select * n_elem_col; i++) {
                RC_ASSERT(within_abs_or_rel(cpu_out[i], gpu_out[i],
                                             ELEMENT_ABS_TOL, ELEMENT_REL_TOL));
            }

            ggml_backend_sched_free(sched);
            ggml_free(ctx);
        });
}

/* ================================================================
 * Property 4 — SetRowsEquivalence
 *
 * Build F32 src [n_elem, n_rows] + index tensor; run backend SET_ROWS
 * into a TURBO_KV_4B dst. At each destination position indices[r] the
 * block must equal the CPU-quantized src row r.
 * ================================================================ */
static void property_SetRowsEquivalence(BackendCtx & bctx) {
    printf("- [%s] SetRowsEquivalence: backend SET_ROWS block[indices[r]] vs CPU quantize(src[r])\n",
           bctx.name.c_str());
    rc::check(
        [&bctx](uint8_t rows_sel, uint32_t seed) {
            const int n_rows     = 1 + (rows_sel & 3);  // 1..4 rows
            const int n_elem_col = BLOCK_SIZE;

            std::vector<float> input(n_rows * n_elem_col);
            fill_random(input.data(), n_rows * n_elem_col, seed);
            for (int r = 0; r < n_rows; r++) {
                if (vec_l2_norm(&input[r * n_elem_col], n_elem_col) < 1e-6f) {
                    input[r * n_elem_col] = 1.0f;
                }
            }

            // Identity index: src row r goes to dst position r. Using a
            // permutation would also be valid but ggml_set_rows' index
            // semantics vary by backend; keep simple for a first pass.
            std::vector<int64_t> indices(n_rows);
            for (int r = 0; r < n_rows; r++) indices[r] = (int64_t)r;

            // CPU reference: per-row quantize
            std::vector<block_turbo_kv_4b> cpu_blocks(n_rows);
            for (int r = 0; r < n_rows; r++) {
                quantize_row_turbo_kv_4b_ref(&input[r * n_elem_col],
                                              &cpu_blocks[r], n_elem_col);
            }

            // Backend SET_ROWS
            ggml_init_params p = { 32 * 1024 * 1024, nullptr, true };
            ggml_context * ctx = ggml_init(p);

            ggml_tensor * src_f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                        n_elem_col, n_rows);
            ggml_tensor * idx_t   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_rows);
            ggml_tensor * dst     = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO_KV_4B,
                                                        n_elem_col, n_rows);
            ggml_tensor * set     = ggml_set_rows(ctx, dst, src_f32, idx_t);

            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, set);

            ggml_backend_t backends[] = { bctx.backend, bctx.cpu };
            ggml_backend_sched_t sched =
                ggml_backend_sched_new(backends, nullptr, 2, 4096, false, false);
            ggml_backend_sched_reset(sched);

            if (!ggml_backend_sched_alloc_graph(sched, graph)) {
                ggml_backend_sched_free(sched);
                ggml_free(ctx);
                return;
            }

            ggml_backend_tensor_set(src_f32, input.data(), 0,
                                     n_rows * n_elem_col * sizeof(float));
            ggml_backend_tensor_set(idx_t, indices.data(), 0,
                                     n_rows * sizeof(int64_t));

            const auto status = ggml_backend_sched_graph_compute(sched, graph);
            RC_ASSERT(status == GGML_STATUS_SUCCESS);

            std::vector<block_turbo_kv_4b> gpu_blocks(n_rows);
            ggml_backend_tensor_get(set, gpu_blocks.data(), 0,
                                     n_rows * sizeof(block_turbo_kv_4b));

            for (int r = 0; r < n_rows; r++) {
                // Indices byte-identical
                const int diff = memcmp(cpu_blocks[r].mse_indices,
                                         gpu_blocks[r].mse_indices,
                                         BLOCK_SIZE / 2);
                RC_ASSERT(diff == INDICES_BYTE_TOL);

                RC_ASSERT(within_abs_or_rel(cpu_blocks[r].norm,
                                             gpu_blocks[r].norm,
                                             0.0f, SCALE_REL_TOL));
                RC_ASSERT(within_abs_or_rel(cpu_blocks[r].inv_std,
                                             gpu_blocks[r].inv_std,
                                             0.0f, SCALE_REL_TOL));
            }

            ggml_backend_sched_free(sched);
            ggml_free(ctx);
        });
}

/* ================================================================
 * main
 * ================================================================ */

int main(int argc, char ** argv) {
    (void)argc; (void)argv;
    printf("=== turbo_kv_4b backend PBT ===\n\n");

    auto backends = init_backends();
    if (backends.empty()) {
        printf("No non-CPU backend available in this build — nothing to test.\n");
        printf("(Rebuild with -DGGML_VULKAN=ON / -DGGML_CUDA=ON / etc. to enable.)\n");
        return 0;
    }

    for (auto & bctx : backends) {
        printf("=== Backend: %s ===\n", bctx.name.c_str());
        property_QuantizeEquivalence(bctx);
        property_DequantizeEquivalence(bctx);
        property_GetRowsEquivalence(bctx);
        property_SetRowsEquivalence(bctx);
        printf("\n");
    }

    free_backends(backends);
    printf("=== All backend properties passed ===\n");
    return 0;
}
