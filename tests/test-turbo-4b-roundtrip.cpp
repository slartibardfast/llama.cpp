/*
 * test-turbo-4b-roundtrip.cpp -- CPU reference validation for TURBO_4B weight quantization
 *
 * Written BEFORE the implementation (test-first). Tests define correctness:
 *   1. Block struct sizes
 *   2. Roundtrip RMSE for both block sizes (128, 64)
 *   3. Cosine similarity preservation
 *   4. Zero vector handling (no NaN/Inf)
 *   5. Outlier handling
 *   6. Codebook index coverage (all 16 used)
 *   7. Cross-blocksize consistency
 *   8. Quality comparison vs Q4_0
 */

#include "ggml-turbo-kv.h"
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

/* Deterministic PRNG (same as test-turbo-kv-vulkan.cpp) */
static float det_rand(uint32_t * state) {
    *state = *state * 1103515245u + 12345u;
    return ((float)(*state >> 16) / 32768.0f) - 1.0f;
}

/* Gaussian-ish via Box-Muller on our deterministic PRNG */
static float det_gauss(uint32_t * state) {
    float u1 = (det_rand(state) + 1.0f) * 0.5f;  /* [0,1) */
    float u2 = (det_rand(state) + 1.0f) * 0.5f;
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307f * u2);
}

static float vec_norm(const float * v, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += v[i] * v[i];
    return sqrtf(s);
}

static float vec_dot(const float * a, const float * b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static float vec_rmse(const float * a, const float * b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return sqrtf(s / n);
}

static bool has_nan_inf(const float * v, int n) {
    for (int i = 0; i < n; i++) {
        if (std::isnan(v[i]) || std::isinf(v[i])) return true;
    }
    return false;
}

/* ================================================================
 * Test 1: Block struct sizes
 * ================================================================ */
static bool test_struct_sizes() {
    fprintf(stderr, "\n=== Test 1: Block struct sizes ===\n");
    bool pass = true;

    if (sizeof(block_turbo_4b) != TURBO_4B_BYTES) {
        fprintf(stderr, "  FAIL: sizeof(block_turbo_4b) = %zu, expected %d\n",
            sizeof(block_turbo_4b), TURBO_4B_BYTES);
        pass = false;
    } else {
        fprintf(stderr, "  block_turbo_4b: %zu bytes (128 elems, 4.25 bpw) PASS\n",
            sizeof(block_turbo_4b));
    }

    if (sizeof(block_turbo_4b_s) != TURBO_4B_S_BYTES) {
        fprintf(stderr, "  FAIL: sizeof(block_turbo_4b_s) = %zu, expected %d\n",
            sizeof(block_turbo_4b_s), TURBO_4B_S_BYTES);
        pass = false;
    } else {
        fprintf(stderr, "  block_turbo_4b_s: %zu bytes (64 elems, 4.5 bpw) PASS\n",
            sizeof(block_turbo_4b_s));
    }

    /* Verify type system */
    if (ggml_blck_size(GGML_TYPE_TURBO_4B) != 128) {
        fprintf(stderr, "  FAIL: blck_size(TURBO_4B) = %d, expected 128\n",
            (int)ggml_blck_size(GGML_TYPE_TURBO_4B));
        pass = false;
    }
    if (ggml_blck_size(GGML_TYPE_TURBO_4B_S) != 64) {
        fprintf(stderr, "  FAIL: blck_size(TURBO_4B_S) = %d, expected 64\n",
            (int)ggml_blck_size(GGML_TYPE_TURBO_4B_S));
        pass = false;
    }
    if (strcmp(ggml_type_name(GGML_TYPE_TURBO_4B), "turbo_4b") != 0) {
        fprintf(stderr, "  FAIL: type_name(TURBO_4B) = '%s'\n",
            ggml_type_name(GGML_TYPE_TURBO_4B));
        pass = false;
    }

    if (pass) fprintf(stderr, "  All struct/type checks PASS\n");
    return pass;
}

/* ================================================================
 * Test 2: Roundtrip RMSE (block_size=128)
 * ================================================================ */
static bool test_roundtrip_128() {
    fprintf(stderr, "\n=== Test 2: Roundtrip RMSE (block_size=128) ===\n");
    bool all_pass = true;

    struct { const char * name; int dim; float scale; float rel_threshold; } cases[] = {
        {"gauss_128",   128, 1.0f, 0.10f},
        {"gauss_256",   256, 1.0f, 0.10f},
        {"gauss_512",   512, 1.0f, 0.10f},
        {"small_128",   128, 0.01f, 0.10f},
        {"large_128",   128, 100.0f, 0.10f},
    };

    for (auto & tc : cases) {
        std::vector<float> input(tc.dim), output(tc.dim);
        uint32_t seed = 12345;
        for (int i = 0; i < tc.dim; i++) input[i] = det_gauss(&seed) * tc.scale;

        int nb = tc.dim / TURBO_4B_BLOCK_SIZE;
        std::vector<block_turbo_4b> blocks(nb);
        quantize_row_turbo_4b_ref(input.data(), blocks.data(), tc.dim);
        dequantize_row_turbo_4b(blocks.data(), output.data(), tc.dim);

        float rmse = vec_rmse(input.data(), output.data(), tc.dim);
        float norm = vec_norm(input.data(), tc.dim);
        float rel_rmse = rmse / fmaxf(norm / sqrtf((float)tc.dim), 1e-10f);

        bool pass = rel_rmse < tc.rel_threshold;
        fprintf(stderr, "  %-16s: RMSE=%.6f norm=%.4f rel=%.4f | %s\n",
            tc.name, rmse, norm, rel_rmse, pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }
    return all_pass;
}

/* ================================================================
 * Test 3: Roundtrip RMSE (block_size=64)
 * ================================================================ */
static bool test_roundtrip_64() {
    fprintf(stderr, "\n=== Test 3: Roundtrip RMSE (block_size=64) ===\n");
    bool all_pass = true;

    struct { const char * name; int dim; float rel_threshold; } cases[] = {
        {"gauss_64",   64,  0.10f},  /* smaller block = slightly higher error */
        {"gauss_128",  128, 0.10f},
        {"gauss_256",  256, 0.10f},
    };

    for (auto & tc : cases) {
        std::vector<float> input(tc.dim), output(tc.dim);
        uint32_t seed = 54321;
        for (int i = 0; i < tc.dim; i++) input[i] = det_gauss(&seed);

        int nb = tc.dim / TURBO_4B_S_BLOCK_SIZE;
        std::vector<block_turbo_4b_s> blocks(nb);
        quantize_row_turbo_4b_s_ref(input.data(), blocks.data(), tc.dim);
        dequantize_row_turbo_4b_s(blocks.data(), output.data(), tc.dim);

        float rmse = vec_rmse(input.data(), output.data(), tc.dim);
        float norm = vec_norm(input.data(), tc.dim);
        float rel_rmse = rmse / fmaxf(norm / sqrtf((float)tc.dim), 1e-10f);

        bool pass = rel_rmse < tc.rel_threshold;
        fprintf(stderr, "  %-16s: RMSE=%.6f norm=%.4f rel=%.4f | %s\n",
            tc.name, rmse, norm, rel_rmse, pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }
    return all_pass;
}

/* ================================================================
 * Test 4: Cosine similarity preservation
 * ================================================================ */
static bool test_cosine_sim() {
    fprintf(stderr, "\n=== Test 4: Cosine similarity preservation ===\n");
    bool all_pass = true;

    for (int block_size : {64, 128}) {
        const int dim = block_size * 2;
        std::vector<float> input(dim), output(dim);
        uint32_t seed = 99999;
        for (int i = 0; i < dim; i++) input[i] = det_gauss(&seed);

        if (block_size == 128) {
            int nb = dim / 128;
            std::vector<block_turbo_4b> blocks(nb);
            quantize_row_turbo_4b_ref(input.data(), blocks.data(), dim);
            dequantize_row_turbo_4b(blocks.data(), output.data(), dim);
        } else {
            int nb = dim / 64;
            std::vector<block_turbo_4b_s> blocks(nb);
            quantize_row_turbo_4b_s_ref(input.data(), blocks.data(), dim);
            dequantize_row_turbo_4b_s(blocks.data(), output.data(), dim);
        }

        float dot_ab = vec_dot(input.data(), output.data(), dim);
        float norm_a = vec_norm(input.data(), dim);
        float norm_b = vec_norm(output.data(), dim);
        float cos_sim = dot_ab / (norm_a * norm_b + 1e-10f);

        bool pass = cos_sim > 0.99f;
        fprintf(stderr, "  bs=%d dim=%d: cos=%.6f | %s\n",
            block_size, dim, cos_sim, pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }
    return all_pass;
}

/* ================================================================
 * Test 5: Zero vector handling
 * ================================================================ */
static bool test_zero_vector() {
    fprintf(stderr, "\n=== Test 5: Zero vector handling ===\n");
    bool pass = true;

    /* block_size=128 */
    {
        std::vector<float> input(128, 0.0f), output(128);
        block_turbo_4b block;
        quantize_row_turbo_4b_ref(input.data(), &block, 128);
        dequantize_row_turbo_4b(&block, output.data(), 128);
        if (has_nan_inf(output.data(), 128)) {
            fprintf(stderr, "  bs=128: NaN/Inf in output FAIL\n");
            pass = false;
        } else {
            fprintf(stderr, "  bs=128: no NaN/Inf PASS\n");
        }
    }

    /* block_size=64 */
    {
        std::vector<float> input(64, 0.0f), output(64);
        block_turbo_4b_s block;
        quantize_row_turbo_4b_s_ref(input.data(), &block, 64);
        dequantize_row_turbo_4b_s(&block, output.data(), 64);
        if (has_nan_inf(output.data(), 64)) {
            fprintf(stderr, "  bs=64: NaN/Inf in output FAIL\n");
            pass = false;
        } else {
            fprintf(stderr, "  bs=64: no NaN/Inf PASS\n");
        }
    }

    return pass;
}

/* ================================================================
 * Test 6: Outlier handling (single 100x outlier)
 * ================================================================ */
static bool test_outlier() {
    fprintf(stderr, "\n=== Test 6: Outlier handling ===\n");

    const int dim = 128;
    std::vector<float> input(dim), output(dim);
    uint32_t seed = 77777;
    for (int i = 0; i < dim; i++) input[i] = det_gauss(&seed);
    input[0] = 100.0f;  /* single 100x outlier */

    block_turbo_4b block;
    quantize_row_turbo_4b_ref(input.data(), &block, dim);
    dequantize_row_turbo_4b(&block, output.data(), dim);

    if (has_nan_inf(output.data(), dim)) {
        fprintf(stderr, "  NaN/Inf in output FAIL\n");
        return false;
    }

    float rmse = vec_rmse(input.data(), output.data(), dim);
    float norm = vec_norm(input.data(), dim);
    float rel_rmse = rmse / fmaxf(norm / sqrtf((float)dim), 1e-10f);

    /* RHT should spread the outlier energy, so rel_rmse should be reasonable */
    bool pass = rel_rmse < 0.15f;
    fprintf(stderr, "  outlier: RMSE=%.4f norm=%.4f rel=%.4f | %s\n",
        rmse, norm, rel_rmse, pass ? "PASS" : "FAIL");
    return pass;
}

/* ================================================================
 * Test 7: Codebook index coverage
 * ================================================================ */
static bool test_codebook_coverage() {
    fprintf(stderr, "\n=== Test 7: Codebook index coverage ===\n");

    const int n_blocks = 100;
    const int dim = 128 * n_blocks;
    std::vector<float> input(dim);
    uint32_t seed = 11111;
    for (int i = 0; i < dim; i++) input[i] = det_gauss(&seed);

    std::vector<block_turbo_4b> blocks(n_blocks);
    quantize_row_turbo_4b_ref(input.data(), blocks.data(), dim);

    int counts[16] = {};
    for (int b = 0; b < n_blocks; b++) {
        for (int i = 0; i < 64; i++) {
            uint8_t byte = blocks[b].qs[i];
            counts[byte & 0x0F]++;
            counts[(byte >> 4) & 0x0F]++;
        }
    }

    int zero_count = 0;
    for (int c = 0; c < 16; c++) {
        if (counts[c] == 0) zero_count++;
    }

    bool pass = zero_count == 0;
    fprintf(stderr, "  indices used: %d/16 | %s\n", 16 - zero_count, pass ? "PASS" : "FAIL");
    if (!pass) {
        for (int c = 0; c < 16; c++) {
            fprintf(stderr, "    codebook[%2d]: %6d hits\n", c, counts[c]);
        }
    }
    return pass;
}

/* ================================================================
 * Test 8: Cross-blocksize consistency
 * ================================================================ */
static bool test_cross_blocksize() {
    fprintf(stderr, "\n=== Test 8: Cross-blocksize consistency ===\n");

    /* Quantize 256 elements as 2x128 vs 4x64 — RMSE should be comparable */
    const int dim = 256;
    std::vector<float> input(dim), out_128(dim), out_64(dim);
    uint32_t seed = 33333;
    for (int i = 0; i < dim; i++) input[i] = det_gauss(&seed);

    /* 2 blocks of 128 */
    std::vector<block_turbo_4b> blk128(2);
    quantize_row_turbo_4b_ref(input.data(), blk128.data(), dim);
    dequantize_row_turbo_4b(blk128.data(), out_128.data(), dim);

    /* 4 blocks of 64 */
    std::vector<block_turbo_4b_s> blk64(4);
    quantize_row_turbo_4b_s_ref(input.data(), blk64.data(), dim);
    dequantize_row_turbo_4b_s(blk64.data(), out_64.data(), dim);

    float rmse_128 = vec_rmse(input.data(), out_128.data(), dim);
    float rmse_64  = vec_rmse(input.data(), out_64.data(), dim);
    float norm = vec_norm(input.data(), dim);

    /* Both should be reasonable; 128 should be slightly better (lower overhead) */
    float ratio = rmse_64 / fmaxf(rmse_128, 1e-10f);
    bool pass = ratio < 2.0f && ratio > 0.5f;
    fprintf(stderr, "  RMSE(128)=%.6f RMSE(64)=%.6f ratio=%.2f norm=%.4f | %s\n",
        rmse_128, rmse_64, ratio, norm, pass ? "PASS" : "FAIL");
    return pass;
}

/* ================================================================
 * Test 9: vec_dot correctness
 * ================================================================ */
static bool test_vec_dot() {
    fprintf(stderr, "\n=== Test 9: vec_dot correctness ===\n");
    bool all_pass = true;

    for (int dim : {128, 256, 512}) {
        std::vector<float> weights(dim), activation(dim), dequant_w(dim);
        uint32_t seed = 44444 + dim;
        for (int i = 0; i < dim; i++) weights[i] = det_gauss(&seed);
        for (int i = 0; i < dim; i++) activation[i] = det_gauss(&seed);

        /* Quantize weights */
        int nb = dim / TURBO_4B_BLOCK_SIZE;
        std::vector<block_turbo_4b> blocks(nb);
        quantize_row_turbo_4b_ref(weights.data(), blocks.data(), dim);

        /* Reference: dequant to float, then dot */
        dequantize_row_turbo_4b(blocks.data(), dequant_w.data(), dim);
        float ref_dot = vec_dot(dequant_w.data(), activation.data(), dim);

        /* vec_dot: direct quantized dot product */
        float vd_result = 0;
        ggml_vec_dot_turbo_4b_f32(dim, &vd_result, 0,
            blocks.data(), 0,
            activation.data(), 0, 1);

        float err = fabsf(ref_dot - vd_result) / fmaxf(fabsf(ref_dot), 1e-10f);
        bool pass = err < 1e-4f;
        fprintf(stderr, "  dim=%d: ref=%.6f vec_dot=%.6f rel_err=%.2e | %s\n",
            dim, ref_dot, vd_result, err, pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }
    return all_pass;
}

/* ================================================================
 * Test 10: Full bitrate ladder roundtrip (2B/3B/5B)
 * ================================================================ */
static bool test_bitrate_ladder() {
    fprintf(stderr, "\n=== Test 10: Full bitrate ladder roundtrip ===\n");
    bool all_pass = true;
    const int dim = 256; /* 2 blocks of 128 */

    std::vector<float> input(dim), output(dim);
    uint32_t seed = 55555;
    for (int i = 0; i < dim; i++) input[i] = det_gauss(&seed);

    /* 2-bit */
    {
        std::vector<block_turbo_2b> blk(dim / TURBO_2B_BLOCK_SIZE);
        quantize_row_turbo_2b_ref(input.data(), blk.data(), dim);
        dequantize_row_turbo_2b(blk.data(), output.data(), dim);
        float rmse = vec_rmse(input.data(), output.data(), dim);
        float norm = vec_norm(input.data(), dim);
        float rel = rmse / fmaxf(norm / sqrtf((float)dim), 1e-10f);
        bool pass = rel < 0.60f; /* 2-bit is coarse — 4 levels */
        fprintf(stderr, "  2-bit: RMSE=%.4f rel=%.4f | %s\n", rmse, rel, pass ? "PASS" : "FAIL");
        all_pass &= pass;

        /* vec_dot */
        std::vector<float> act(dim), dq(dim);
        for (int i = 0; i < dim; i++) act[i] = det_gauss(&seed);
        dequantize_row_turbo_2b(blk.data(), dq.data(), dim);
        float ref_d = vec_dot(dq.data(), act.data(), dim);
        float vd = 0;
        ggml_vec_dot_turbo_2b_f32(dim, &vd, 0, blk.data(), 0, act.data(), 0, 1);
        float err = fabsf(ref_d - vd) / fmaxf(fabsf(ref_d), 1e-10f);
        bool vpass = err < 1e-4f;
        fprintf(stderr, "  2-bit vec_dot: err=%.2e | %s\n", err, vpass ? "PASS" : "FAIL");
        all_pass &= vpass;
    }

    /* 3-bit */
    {
        std::vector<block_turbo_3b> blk(dim / TURBO_3B_BLOCK_SIZE);
        quantize_row_turbo_3b_ref(input.data(), blk.data(), dim);
        dequantize_row_turbo_3b(blk.data(), output.data(), dim);
        float rmse = vec_rmse(input.data(), output.data(), dim);
        float norm = vec_norm(input.data(), dim);
        float rel = rmse / fmaxf(norm / sqrtf((float)dim), 1e-10f);
        bool pass = rel < 0.25f; /* 3-bit — 8 levels */
        fprintf(stderr, "  3-bit: RMSE=%.4f rel=%.4f | %s\n", rmse, rel, pass ? "PASS" : "FAIL");
        all_pass &= pass;

        std::vector<float> act(dim), dq(dim);
        for (int i = 0; i < dim; i++) act[i] = det_gauss(&seed);
        dequantize_row_turbo_3b(blk.data(), dq.data(), dim);
        float ref_d = vec_dot(dq.data(), act.data(), dim);
        float vd = 0;
        ggml_vec_dot_turbo_3b_f32(dim, &vd, 0, blk.data(), 0, act.data(), 0, 1);
        float err = fabsf(ref_d - vd) / fmaxf(fabsf(ref_d), 1e-10f);
        bool vpass = err < 1e-4f;
        fprintf(stderr, "  3-bit vec_dot: err=%.2e | %s\n", err, vpass ? "PASS" : "FAIL");
        all_pass &= vpass;
    }

    /* 5-bit */
    {
        std::vector<block_turbo_5b> blk(dim / TURBO_5B_BLOCK_SIZE);
        quantize_row_turbo_5b_ref(input.data(), blk.data(), dim);
        dequantize_row_turbo_5b(blk.data(), output.data(), dim);
        float rmse = vec_rmse(input.data(), output.data(), dim);
        float norm = vec_norm(input.data(), dim);
        float rel = rmse / fmaxf(norm / sqrtf((float)dim), 1e-10f);
        bool pass = rel < 0.06f; /* 5-bit should be very good */
        fprintf(stderr, "  5-bit: RMSE=%.4f rel=%.4f | %s\n", rmse, rel, pass ? "PASS" : "FAIL");
        all_pass &= pass;

        std::vector<float> act(dim), dq(dim);
        for (int i = 0; i < dim; i++) act[i] = det_gauss(&seed);
        dequantize_row_turbo_5b(blk.data(), dq.data(), dim);
        float ref_d = vec_dot(dq.data(), act.data(), dim);
        float vd = 0;
        ggml_vec_dot_turbo_5b_f32(dim, &vd, 0, blk.data(), 0, act.data(), 0, 1);
        float err = fabsf(ref_d - vd) / fmaxf(fabsf(ref_d), 1e-10f);
        bool vpass = err < 1e-4f;
        fprintf(stderr, "  5-bit vec_dot: err=%.2e | %s\n", err, vpass ? "PASS" : "FAIL");
        all_pass &= vpass;
    }

    return all_pass;
}

/* ================================================================
 * Main
 * ================================================================ */
int main() {
    fprintf(stderr, "TURBO Weight Quantization — Full Bitrate Ladder Tests\n");
    fprintf(stderr, "=====================================================\n");

    int pass = 0, fail = 0;

    auto run = [&](bool (*fn)(), const char * name) {
        if (fn()) { pass++; }
        else      { fail++; fprintf(stderr, "  >>> %s FAILED <<<\n", name); }
    };

    /* Test 1 always passes (struct sizes are compile-time) */
    run(test_struct_sizes,       "struct_sizes");

    /* Tests 2-8 require quantize/dequant implementation */
    run(test_roundtrip_128,      "roundtrip_128");
    run(test_roundtrip_64,       "roundtrip_64");
    run(test_cosine_sim,         "cosine_sim");
    run(test_zero_vector,        "zero_vector");
    run(test_outlier,            "outlier");
    run(test_codebook_coverage,  "codebook_coverage");
    run(test_cross_blocksize,    "cross_blocksize");

    /* Test 9 requires vec_dot implementation */
    run(test_vec_dot,            "vec_dot");

    /* Test 10: full bitrate ladder (2B/3B/5B) */
    run(test_bitrate_ladder,     "bitrate_ladder");

    fprintf(stderr, "\n================================================\n");
    fprintf(stderr, "Results: %d passed, %d failed\n", pass, fail);

    return fail > 0 ? 1 : 0;
}
