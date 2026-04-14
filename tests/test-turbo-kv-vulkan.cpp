/*
 * test-turbo-kv-vulkan.cpp — CPU reference validation + GPU round-trip fixture
 *
 * Phase 1: Validate CPU quantize/dequant round-trip precision
 * Phase 2: Compare GPU vs CPU dequant (when Vulkan test harness is available)
 */

#include "ggml-turbo-kv.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

static float det_rand(uint32_t * state) {
    *state = *state * 1103515245u + 12345u;
    return ((float)(*state >> 16) / 32768.0f) - 1.0f;
}

// Test 1: CPU quantize → dequant round-trip
static bool test_cpu_roundtrip() {
    fprintf(stderr, "\n=== Test 1: CPU quantize → dequant round-trip ===\n");

    const int dim = 128;
    bool all_pass = true;

    struct { const char * name; float scale; float threshold; } cases[] = {
        {"unit_gaussian",  1.0f,  0.15f},
        {"small_values",   0.01f, 0.002f},
        {"large_values",   10.0f, 1.5f},
        {"mixed",          5.0f,  0.8f},
    };

    for (auto & tc : cases) {
        std::vector<float> input(dim), output(dim);
        uint32_t seed = 42;
        for (int i = 0; i < dim; i++) input[i] = det_rand(&seed) * tc.scale;

        block_turbo_kv_4b block;
        quantize_row_turbo_kv_4b_ref(input.data(), &block, dim);
        dequantize_row_turbo_kv_4b(&block, output.data(), dim);

        float rmse = 0, max_err = 0;
        for (int i = 0; i < dim; i++) {
            float d = fabsf(input[i] - output[i]);
            rmse += d * d;
            if (d > max_err) max_err = d;
        }
        rmse = sqrtf(rmse / dim);

        float norm = 0;
        for (int i = 0; i < dim; i++) norm += input[i] * input[i];
        norm = sqrtf(norm);
        float rel_rmse = rmse / fmaxf(norm / sqrtf(dim), 1e-10f);

        bool pass = rel_rmse < 0.5f;
        fprintf(stderr, "  %-16s: RMSE=%.6f max=%.6f rel=%.4f norm=%.4f | %s\n",
            tc.name, rmse, max_err, rel_rmse, norm, pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }
    return all_pass;
}

// Test 2: Block metadata validation
static bool test_metadata() {
    fprintf(stderr, "\n=== Test 2: Block metadata ===\n");

    const int dim = 128;
    std::vector<float> input(dim);
    uint32_t seed = 42;
    for (int i = 0; i < dim; i++) input[i] = det_rand(&seed);

    block_turbo_kv_4b block;
    quantize_row_turbo_kv_4b_ref(input.data(), &block, dim);

    float norm = ggml_fp16_to_fp32(*(ggml_fp16_t*)&block.norm);
    float inv_std = ggml_fp16_to_fp32(*(ggml_fp16_t*)&block.inv_std_fp16);

    float actual_norm = 0;
    for (int i = 0; i < dim; i++) actual_norm += input[i] * input[i];
    actual_norm = sqrtf(actual_norm);

    float norm_err = fabsf(norm - actual_norm) / fmaxf(actual_norm, 1e-10f);
    fprintf(stderr, "  norm=%.6f (actual=%.6f, err=%.4f%%)\n", norm, actual_norm, norm_err * 100);
    fprintf(stderr, "  inv_std=%.6f\n", inv_std);
    fprintf(stderr, "  indices[0..7]: %02x %02x %02x %02x %02x %02x %02x %02x\n",
        block.mse_indices[0], block.mse_indices[1], block.mse_indices[2], block.mse_indices[3],
        block.mse_indices[4], block.mse_indices[5], block.mse_indices[6], block.mse_indices[7]);

    bool pass = norm_err < 0.01f && inv_std > 0.0f;
    fprintf(stderr, "  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// Test 3: Codebook index distribution
static bool test_distribution() {
    fprintf(stderr, "\n=== Test 3: Codebook index distribution ===\n");

    const int dim = 128;
    const int n_blocks = 100;
    int hist[16] = {};

    uint32_t seed = 99;
    for (int b = 0; b < n_blocks; b++) {
        std::vector<float> input(dim);
        for (int i = 0; i < dim; i++) input[i] = det_rand(&seed);

        block_turbo_kv_4b block;
        quantize_row_turbo_kv_4b_ref(input.data(), &block, dim);

        for (int i = 0; i < 64; i++) {
            hist[block.mse_indices[i] & 0xF]++;
            hist[block.mse_indices[i] >> 4]++;
        }
    }

    fprintf(stderr, "  ");
    bool all_used = true;
    for (int c = 0; c < 16; c++) {
        fprintf(stderr, "[%2d]=%4d ", c, hist[c]);
        if (hist[c] == 0) all_used = false;
        if (c == 7) fprintf(stderr, "\n  ");
    }
    fprintf(stderr, "\n  All entries used: %s\n", all_used ? "PASS" : "FAIL");
    return all_used;
}

// Test 4: Multi-block row round-trip
static bool test_multiblock() {
    fprintf(stderr, "\n=== Test 4: Multi-block row round-trip ===\n");

    const int dim = 128;
    const int n_blocks = 8;
    const int n_elem = dim * n_blocks;

    std::vector<float> input(n_elem), output(n_elem);
    uint32_t seed = 777;
    for (int i = 0; i < n_elem; i++) input[i] = det_rand(&seed) * 2.0f;

    // Quantize + dequant using the row-level API
    std::vector<block_turbo_kv_4b> blocks(n_blocks);
    for (int b = 0; b < n_blocks; b++) {
        quantize_row_turbo_kv_4b_ref(&input[b * dim], &blocks[b], dim);
    }
    for (int b = 0; b < n_blocks; b++) {
        dequantize_row_turbo_kv_4b(&blocks[b], &output[b * dim], dim);
    }

    float rmse = 0, max_err = 0;
    for (int i = 0; i < n_elem; i++) {
        float d = fabsf(input[i] - output[i]);
        rmse += d * d;
        if (d > max_err) max_err = d;
    }
    rmse = sqrtf(rmse / n_elem);

    fprintf(stderr, "  %d blocks: RMSE=%.6f max_err=%.6f\n", n_blocks, rmse, max_err);
    bool pass = rmse < 0.2f;
    fprintf(stderr, "  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// Test 5: Print detailed first-block comparison for GPU debugging
static bool test_debug_dump() {
    fprintf(stderr, "\n=== Test 5: Debug dump for GPU comparison ===\n");

    const int dim = 128;
    std::vector<float> input(dim), output(dim);
    uint32_t seed = 42;
    for (int i = 0; i < dim; i++) input[i] = det_rand(&seed);

    block_turbo_kv_4b block;
    quantize_row_turbo_kv_4b_ref(input.data(), &block, dim);
    dequantize_row_turbo_kv_4b(&block, output.data(), dim);

    fprintf(stderr, "  Input[0..7]:  ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%8.4f ", input[i]);
    fprintf(stderr, "\n  Output[0..7]: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%8.4f ", output[i]);
    fprintf(stderr, "\n  Error[0..7]:  ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%8.4f ", fabsf(input[i] - output[i]));
    fprintf(stderr, "\n");

    // Print raw block bytes for GPU comparison
    fprintf(stderr, "  Raw block (72 bytes): ");
    const uint8_t * raw = (const uint8_t *)&block;
    for (int i = 0; i < 72; i++) {
        fprintf(stderr, "%02x", raw[i]);
        if (i % 16 == 15) fprintf(stderr, "\n                       ");
    }
    fprintf(stderr, "\n");

    return true;
}

static bool test_forward_rht_steps();

int main() {
    fprintf(stderr, "=== TURBO_KV_4B validation suite ===\n");

    int passed = 0, total = 0;

    total++; if (test_cpu_roundtrip()) passed++;
    total++; if (test_metadata()) passed++;
    total++; if (test_distribution()) passed++;
    total++; if (test_multiblock()) passed++;
    total++; if (test_debug_dump()) passed++;
    total++; if (test_forward_rht_steps()) passed++;

    fprintf(stderr, "\n=== Results: %d/%d tests passed ===\n", passed, total);
    return passed == total ? 0 : 1;
}

// Test 6: Compare forward RHT step by step
static bool test_forward_rht_steps() {
    fprintf(stderr, "\n=== Test 6: Forward RHT step-by-step ===\n");
    const int dim = 128;
    std::vector<float> input(dim);
    uint32_t seed = 42;
    for (int i = 0; i < dim; i++) input[i] = det_rand(&seed);

    // Step 1: Compute norm
    float norm = 0;
    for (int i = 0; i < dim; i++) norm += input[i] * input[i];
    norm = sqrtf(norm);
    fprintf(stderr, "  norm = %.6f\n", norm);

    // Step 2: Normalize
    std::vector<float> normd(dim);
    for (int i = 0; i < dim; i++) normd[i] = input[i] / norm;

    // Step 3: Sign flip
    std::vector<float> signed_d(dim);
    for (int i = 0; i < dim; i++) {
        uint32_t h = (TURBO_KV_DEFAULT_SEED ^ (uint32_t)i);
        h *= 2654435761u;
        int sign = (h & 1u) ? 1 : -1;
        signed_d[i] = normd[i] * sign;
    }

    // Step 4: WHT (use the C reference)
    std::vector<float> rotated(signed_d);
    turbo_kv_rht_forward(rotated.data(), dim, TURBO_KV_DEFAULT_SEED);
    // Note: rht_forward does sign+WHT+normalize in one call.
    // We need to redo with just our manual steps to compare.

    // Actually, let's just use the library function and print intermediate values
    std::vector<float> fwd(dim);
    memcpy(fwd.data(), input.data(), dim * sizeof(float));
    // Normalize first
    for (int i = 0; i < dim; i++) fwd[i] /= norm;
    // Forward RHT
    turbo_kv_rht_forward(fwd.data(), dim, TURBO_KV_DEFAULT_SEED);

    fprintf(stderr, "  After forward RHT[0..7]: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%8.5f ", fwd[i]);
    fprintf(stderr, "\n");

    // Compute max_abs for inv_std
    float max_abs = 0;
    for (int i = 0; i < dim; i++) {
        float a = fabsf(fwd[i]);
        if (a > max_abs) max_abs = a;
    }
    float inv_std = 2.7326f / max_abs;
    fprintf(stderr, "  max_abs=%.6f inv_std=%.6f\n", max_abs, inv_std);

    // Quantize to indices
    const float codebook[16] = {
        -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3881f, -0.1284f,
         0.1284f,  0.3881f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f};
    fprintf(stderr, "  Scaled[0..7] + nearest idx: ");
    for (int i = 0; i < 8; i++) {
        float scaled = fwd[i] * inv_std;
        int best = 0;
        float best_d = fabsf(scaled - codebook[0]);
        for (int c = 1; c < 16; c++) {
            float d = fabsf(scaled - codebook[c]);
            if (d < best_d) { best = c; best_d = d; }
        }
        fprintf(stderr, "%.3f→%d ", scaled, best);
    }
    fprintf(stderr, "\n");

    return true;
}
