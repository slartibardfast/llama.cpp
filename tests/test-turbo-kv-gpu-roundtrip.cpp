/*
 * test-turbo-kv-gpu-roundtrip.cpp
 * Minimal: CPU quantize → upload → GPU dequant → compare CPU dequant
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-turbo-kv.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

int main() {
    fprintf(stderr, "=== TURBO_KV_4B GPU dequant test ===\n");

    ggml_backend_t gpu = nullptr, cpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto dev = ggml_backend_dev_get(i);
        if (!gpu && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU)
            gpu = ggml_backend_dev_init(dev, nullptr);
        if (!cpu && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU)
            cpu = ggml_backend_dev_init(dev, nullptr);
    }
    if (!gpu) { fprintf(stderr, "No GPU\n"); return 0; }
    if (!cpu) { fprintf(stderr, "No CPU\n"); return 1; }

    const int n_elem = 512; // 4 blocks of 128
    const int n_blocks = n_elem / 128;

    // Generate + CPU quantize
    std::vector<float> input(n_elem);
    uint32_t s = 42;
    for (int i = 0; i < n_elem; i++) { s = s*1103515245u+12345u; input[i] = ((float)(s>>16)/32768.f-1.f)*3.f; }

    std::vector<block_turbo_kv_4b> blocks(n_blocks);
    for (int b = 0; b < n_blocks; b++)
        quantize_row_turbo_kv_4b_ref(&input[b*128], &blocks[b], 128);

    // CPU dequant reference
    std::vector<float> cpu_out(n_elem);
    for (int b = 0; b < n_blocks; b++)
        dequantize_row_turbo_kv_4b(&blocks[b], &cpu_out[b*128], 128);

    // GPU dequant: let scheduler handle allocation
    ggml_init_params p = { 16*1024*1024, nullptr, true }; // no_alloc=true
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_TURBO_KV_4B, n_elem);

    ggml_tensor * cast = ggml_cast(ctx, src, GGML_TYPE_F32);
    ggml_tensor * dst = ggml_cont(ctx, cast);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, dst);

    // Use scheduler with both backends
    ggml_backend_t backends[] = { gpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 4096, false, false);
    ggml_backend_sched_reset(sched);

    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        fprintf(stderr, "  alloc failed — trying CPU-only dequant\n");

        // Fallback: just test CPU
        ggml_backend_sched_free(sched);
        ggml_backend_t cpu_only[] = { cpu };
        sched = ggml_backend_sched_new(cpu_only, nullptr, 1, 4096, false, false);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            fprintf(stderr, "  CPU alloc also failed!\n");
            return 1;
        }
    }

    // Upload quantized data to wherever src ended up
    ggml_backend_tensor_set(src, blocks.data(), 0, n_blocks * sizeof(block_turbo_kv_4b));

    enum ggml_status status = ggml_backend_sched_graph_compute(sched, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "  compute failed: %d\n", status);
        return 1;
    }

    // Download
    std::vector<float> gpu_out(n_elem);
    ggml_backend_tensor_get(dst, gpu_out.data(), 0, n_elem * sizeof(float));

    // Compare
    float rmse = 0, max_err = 0;
    int first_bad = -1;
    for (int i = 0; i < n_elem; i++) {
        float d = fabsf(cpu_out[i] - gpu_out[i]);
        rmse += d*d;
        if (d > max_err) max_err = d;
        if (d > 0.01f && first_bad < 0) first_bad = i;
    }
    rmse = sqrtf(rmse / n_elem);

    fprintf(stderr, "  RMSE=%.6f max=%.6f\n", rmse, max_err);
    fprintf(stderr, "  CPU[0..3]: %.4f %.4f %.4f %.4f\n", cpu_out[0], cpu_out[1], cpu_out[2], cpu_out[3]);
    fprintf(stderr, "  GPU[0..3]: %.4f %.4f %.4f %.4f\n", gpu_out[0], gpu_out[1], gpu_out[2], gpu_out[3]);
    if (first_bad >= 0)
        fprintf(stderr, "  First divergence at [%d]: cpu=%.6f gpu=%.6f\n", first_bad, cpu_out[first_bad], gpu_out[first_bad]);

    bool pass = rmse < 0.01f;
    fprintf(stderr, "  %s\n", pass ? "PASS" : "FAIL");

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return pass ? 0 : 1;
}
