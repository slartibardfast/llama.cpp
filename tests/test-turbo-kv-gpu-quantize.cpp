/*
 * test-turbo-kv-gpu-quantize.cpp
 * Compare GPU vs CPU quantize output: same F32 input → compare block bytes.
 * Uses ggml graph: F32 input → CPY → TURBO_KV_4B output
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-turbo-kv.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

int main() {
    fprintf(stderr, "=== TURBO_KV_4B GPU quantize comparison ===\n");

    ggml_backend_t gpu = nullptr, cpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto dev = ggml_backend_dev_get(i);
        if (!gpu && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU)
            gpu = ggml_backend_dev_init(dev, nullptr);
        if (!cpu && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU)
            cpu = ggml_backend_dev_init(dev, nullptr);
    }
    if (!gpu || !cpu) { fprintf(stderr, "Missing backend\n"); return 1; }

    const int n_elem = 128; // single block
    const int n_blocks = 1;

    // Generate test input
    std::vector<float> input(n_elem);
    uint32_t s = 42;
    for (int i = 0; i < n_elem; i++) { s = s*1103515245u+12345u; input[i] = ((float)(s>>16)/32768.f-1.f); }

    // CPU quantize
    block_turbo_kv_4b cpu_block;
    quantize_row_turbo_kv_4b_ref(input.data(), &cpu_block, n_elem);

    // GPU quantize via CPY: F32 → TURBO_KV_4B
    ggml_init_params p = { 16*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elem);
    ggml_tensor * dst_quant = ggml_new_tensor_1d(ctx, GGML_TYPE_TURBO_KV_4B, n_elem);
    ggml_tensor * cpy = ggml_cpy(ctx, src, dst_quant);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, cpy);

    ggml_backend_t backends[] = { gpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 4096, false, false);
    ggml_backend_sched_reset(sched);

    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        fprintf(stderr, "  alloc failed\n"); return 1;
    }

    // Upload input
    ggml_backend_tensor_set(src, input.data(), 0, n_elem * sizeof(float));

    enum ggml_status status = ggml_backend_sched_graph_compute(sched, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "  compute failed: %d\n", status); return 1;
    }

    // Download quantized block
    block_turbo_kv_4b gpu_block;
    ggml_backend_tensor_get(cpy, &gpu_block, 0, sizeof(block_turbo_kv_4b));

    // Compare
    float cpu_norm = ggml_fp16_to_fp32(*(ggml_fp16_t*)&cpu_block.norm);
    float gpu_norm = ggml_fp16_to_fp32(*(ggml_fp16_t*)&gpu_block.norm);
    float cpu_inv = ggml_fp16_to_fp32(*(ggml_fp16_t*)&cpu_block.inv_std_fp16);
    float gpu_inv = ggml_fp16_to_fp32(*(ggml_fp16_t*)&gpu_block.inv_std_fp16);

    fprintf(stderr, "  CPU: norm=%.4f inv_std=%.4f\n", cpu_norm, cpu_inv);
    fprintf(stderr, "  GPU: norm=%.4f inv_std=%.4f\n", gpu_norm, gpu_inv);

    // Compare indices byte by byte
    int idx_diff = 0;
    for (int i = 0; i < 64; i++) {
        if (cpu_block.mse_indices[i] != gpu_block.mse_indices[i]) idx_diff++;
    }
    fprintf(stderr, "  Index bytes different: %d/64\n", idx_diff);

    if (idx_diff > 0) {
        fprintf(stderr, "  CPU indices[0..7]: %02x %02x %02x %02x %02x %02x %02x %02x\n",
            cpu_block.mse_indices[0], cpu_block.mse_indices[1], cpu_block.mse_indices[2], cpu_block.mse_indices[3],
            cpu_block.mse_indices[4], cpu_block.mse_indices[5], cpu_block.mse_indices[6], cpu_block.mse_indices[7]);
        fprintf(stderr, "  GPU indices[0..7]: %02x %02x %02x %02x %02x %02x %02x %02x\n",
            gpu_block.mse_indices[0], gpu_block.mse_indices[1], gpu_block.mse_indices[2], gpu_block.mse_indices[3],
            gpu_block.mse_indices[4], gpu_block.mse_indices[5], gpu_block.mse_indices[6], gpu_block.mse_indices[7]);
    }

    // Now dequant both blocks on CPU and compare
    std::vector<float> cpu_dequant(n_elem), gpu_dequant(n_elem);
    dequantize_row_turbo_kv_4b(&cpu_block, cpu_dequant.data(), n_elem);
    dequantize_row_turbo_kv_4b(&gpu_block, gpu_dequant.data(), n_elem);

    float rmse = 0;
    for (int i = 0; i < n_elem; i++) {
        float d = cpu_dequant[i] - gpu_dequant[i];
        rmse += d*d;
    }
    rmse = sqrtf(rmse / n_elem);
    fprintf(stderr, "  CPU-dequant vs GPU-dequant RMSE: %.6f\n", rmse);

    bool pass = idx_diff == 0 && fabsf(cpu_norm - gpu_norm) < 0.01f;
    fprintf(stderr, "  %s\n", pass ? "PASS" : "FAIL");

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return pass ? 0 : 1;
}
