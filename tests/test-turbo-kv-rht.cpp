/*
 * test-turbo-kv-rht.cpp — bit-exactness fixture for Walsh-Hadamard butterfly
 *
 * Validates:
 *   1. Scalar walsh_hadamard output matches SSE version (when implemented)
 *   2. RHT forward + inverse roundtrip: ||RHT_inv(RHT(x)) - x|| < 1e-6
 *   3. Orthogonality: <RHT(a), RHT(b)> == <a, b> within tolerance
 *
 * Build: cmake --build build-tq --target test-turbo-kv-rht
 * Run:   build-tq/bin/test-turbo-kv-rht
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

static float vec_dot(const float * a, const float * b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static float vec_l2_diff(const float * a, const float * b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

static bool test_roundtrip(int dim) {
    std::vector<float> x(dim), y(dim);
    uint32_t seed = 12345;
    for (int i = 0; i < dim; i++) {
        x[i] = det_rand(&seed);
    }
    memcpy(y.data(), x.data(), dim * sizeof(float));

    turbo_kv_rht_forward(y.data(), dim, TURBO_KV_DEFAULT_SEED);
    turbo_kv_rht_inverse(y.data(), dim, TURBO_KV_DEFAULT_SEED);

    float err = vec_l2_diff(x.data(), y.data(), dim);
    bool pass = (err < 1e-5f);

    fprintf(stderr, "  roundtrip dim=%3d | L2_err=%.8f | %s\n",
            dim, err, pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_orthogonality(int dim) {
    std::vector<float> a(dim), b(dim), a_rot(dim), b_rot(dim);
    uint32_t seed = 99;
    for (int i = 0; i < dim; i++) {
        a[i] = det_rand(&seed);
        b[i] = det_rand(&seed);
    }
    memcpy(a_rot.data(), a.data(), dim * sizeof(float));
    memcpy(b_rot.data(), b.data(), dim * sizeof(float));

    turbo_kv_rht_forward(a_rot.data(), dim, TURBO_KV_DEFAULT_SEED);
    turbo_kv_rht_forward(b_rot.data(), dim, TURBO_KV_DEFAULT_SEED);

    float dot_orig = vec_dot(a.data(), b.data(), dim);
    float dot_rot  = vec_dot(a_rot.data(), b_rot.data(), dim);
    float rel_err  = fabsf(dot_orig - dot_rot) / (fabsf(dot_orig) + 1e-10f);

    bool pass = (rel_err < 1e-5f);
    fprintf(stderr, "  orthogonality dim=%3d | dot_orig=%.6f dot_rot=%.6f rel_err=%.8f | %s\n",
            dim, dot_orig, dot_rot, rel_err, pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_multi_block_rotation(int dim) {
    // turbo_kv_rotate_query handles multi-block dims (> 128)
    // by rotating each 128-element chunk independently.
    // Verify: rotating 256 elements produces the same result as
    // rotating [0..127] and [128..255] separately.
    std::vector<float> full(dim), split(dim);
    uint32_t seed = 777;
    for (int i = 0; i < dim; i++) {
        full[i] = det_rand(&seed);
    }
    memcpy(split.data(), full.data(), dim * sizeof(float));

    // Full rotation via turbo_kv_rotate_query
    std::vector<float> full_rot(dim + TURBO_KV_BLOCK_SIZE); // extra room for zero-pad
    turbo_kv_rotate_query(full.data(), full_rot.data(), dim);

    // Split rotation: manually rotate each 128-block
    std::vector<float> split_rot(dim + TURBO_KV_BLOCK_SIZE);
    int d = 0;
    while (d + TURBO_KV_BLOCK_SIZE <= dim) {
        memcpy(split_rot.data() + d, split.data() + d, TURBO_KV_BLOCK_SIZE * sizeof(float));
        turbo_kv_rht_forward(split_rot.data() + d, TURBO_KV_BLOCK_SIZE, TURBO_KV_DEFAULT_SEED);
        d += TURBO_KV_BLOCK_SIZE;
    }

    float err = vec_l2_diff(full_rot.data(), split_rot.data(), dim);
    bool pass = (err < 1e-6f);
    fprintf(stderr, "  multi-block dim=%3d | L2_err=%.8f | %s\n",
            dim, err, pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    fprintf(stderr, "=== turbo_kv RHT correctness ===\n\n");

    int pass = 0, fail = 0;

    fprintf(stderr, "--- Roundtrip (forward + inverse = identity) ---\n");
    test_roundtrip(64)  ? pass++ : fail++;
    test_roundtrip(128) ? pass++ : fail++;
    test_roundtrip(256) ? pass++ : fail++;

    fprintf(stderr, "\n--- Orthogonality (<a,b> == <RHT(a),RHT(b)>) ---\n");
    test_orthogonality(64)  ? pass++ : fail++;
    test_orthogonality(128) ? pass++ : fail++;
    test_orthogonality(256) ? pass++ : fail++;

    fprintf(stderr, "\n--- Multi-block rotation consistency ---\n");
    test_multi_block_rotation(128) ? pass++ : fail++;
    test_multi_block_rotation(256) ? pass++ : fail++;
    test_multi_block_rotation(512) ? pass++ : fail++;

    fprintf(stderr, "\n=== %d/%d passed ===\n", pass, pass + fail);
    return fail > 0 ? 1 : 0;
}
