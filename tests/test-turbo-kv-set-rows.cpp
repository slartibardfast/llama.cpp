/*
 * test-turbo-kv-set-rows.cpp
 * Compare SET_ROWS vs CPY quantize output for TURBO_KV_4B.
 * CPY is proven correct. If SET_ROWS differs, the addressing is the bug.
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-turbo-kv.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

int main() {
    fprintf(stderr, "=== SET_ROWS vs CPY comparison for TURBO_KV_4B ===\n");

    ggml_backend_t gpu = nullptr, cpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto dev = ggml_backend_dev_get(i);
        if (!gpu && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU)
            gpu = ggml_backend_dev_init(dev, nullptr);
        if (!cpu && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU)
            cpu = ggml_backend_dev_init(dev, nullptr);
    }
    if (!gpu || !cpu) { fprintf(stderr, "Missing backend\n"); return 1; }

    const int n_elem = 128;  // single block
    const int n_rows = 4;    // multiple rows like the real workload

    // Test 1: Single block via CPY (reference)
    fprintf(stderr, "\n--- Test 1: CPY single block ---\n");
    {
        ggml_init_params p = { 16*1024*1024, nullptr, true };
        ggml_context * ctx = ggml_init(p);

        ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elem);
        ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_TURBO_KV_4B, n_elem);
        ggml_tensor * cpy = ggml_cpy(ctx, src, dst);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, cpy);

        ggml_backend_t backends[] = { gpu, cpu };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 4096, false, false);
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, graph);

        // Fill with known data
        std::vector<float> input(n_elem);
        uint32_t s = 42;
        for (int i = 0; i < n_elem; i++) { s = s*1103515245u+12345u; input[i] = ((float)(s>>16)/32768.f-1.f); }
        ggml_backend_tensor_set(src, input.data(), 0, n_elem * sizeof(float));

        ggml_backend_sched_graph_compute(sched, graph);

        block_turbo_kv_4b cpy_block;
        ggml_backend_tensor_get(cpy, &cpy_block, 0, sizeof(block_turbo_kv_4b));

        fprintf(stderr, "  CPY norm=%.6f inv_std=%.6f\n", cpy_block.norm, cpy_block.inv_std);
        fprintf(stderr, "  CPY idx[0..3]: %02x %02x %02x %02x\n",
            cpy_block.mse_indices[0], cpy_block.mse_indices[1], cpy_block.mse_indices[2], cpy_block.mse_indices[3]);

        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }

    // Test 2: Single block via SET_ROWS with identity index [0]
    fprintf(stderr, "\n--- Test 2: SET_ROWS single block (identity index) ---\n");
    {
        ggml_init_params p = { 16*1024*1024, nullptr, true };
        ggml_context * ctx = ggml_init(p);

        // src0: F32 input [n_elem, 1]
        ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_elem, 1);
        // src1: row index [1] with value 0 (identity mapping)
        ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
        ggml_set_input(idx);
        // dst: TURBO_KV_4B cache [n_elem, 1]
        ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO_KV_4B, n_elem, 1);
        ggml_tensor * set = ggml_set_rows(ctx, dst, src0, idx);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, set);

        ggml_backend_t backends[] = { gpu, cpu };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 4096, false, false);
        ggml_backend_sched_reset(sched);

        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            fprintf(stderr, "  alloc failed!\n");
            ggml_backend_sched_free(sched);
            ggml_free(ctx);
            ggml_backend_free(gpu);
            ggml_backend_free(cpu);
            return 1;
        }

        // Fill same input data
        std::vector<float> input(n_elem);
        uint32_t s = 42;
        for (int i = 0; i < n_elem; i++) { s = s*1103515245u+12345u; input[i] = ((float)(s>>16)/32768.f-1.f); }
        ggml_backend_tensor_set(src0, input.data(), 0, n_elem * sizeof(float));

        // Identity index: row 0 → row 0
        int64_t idx_val = 0;
        ggml_backend_tensor_set(idx, &idx_val, 0, sizeof(int64_t));

        ggml_backend_sched_graph_compute(sched, graph);

        block_turbo_kv_4b sr_block;
        ggml_backend_tensor_get(set, &sr_block, 0, sizeof(block_turbo_kv_4b));

        fprintf(stderr, "  SR  norm=%.6f inv_std=%.6f\n", sr_block.norm, sr_block.inv_std);
        fprintf(stderr, "  SR  idx[0..3]: %02x %02x %02x %02x\n",
            sr_block.mse_indices[0], sr_block.mse_indices[1], sr_block.mse_indices[2], sr_block.mse_indices[3]);

        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return 0;
}
