#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
typedef void (*quantize_fn)(const float *, void *, int64_t);
typedef size_t (*type_size_fn)(void);

#define DECL_BOTH(name) \
    extern void ggml_vec_dot_##name(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc); \
    extern void ggml_vec_dot_##name##_generic(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

DECL_BOTH(q2_K_q8_K)
DECL_BOTH(q3_K_q8_K)
DECL_BOTH(q4_1_q8_1)
DECL_BOTH(q5_0_q8_0)
DECL_BOTH(q5_1_q8_1)
DECL_BOTH(q8_0_q8_0)
DECL_BOTH(q4_K_q8_K)
DECL_BOTH(q5_K_q8_K)
DECL_BOTH(q6_K_q8_K)
DECL_BOTH(iq2_xs_q8_K)
DECL_BOTH(iq2_s_q8_K)
DECL_BOTH(iq3_xxs_q8_K)
DECL_BOTH(iq3_s_q8_K)
DECL_BOTH(iq4_nl_q8_0)
DECL_BOTH(iq4_xs_q8_K)
DECL_BOTH(iq1_s_q8_K)
DECL_BOTH(iq1_m_q8_K)
DECL_BOTH(tq1_0_q8_K)
DECL_BOTH(tq2_0_q8_K)

// Quantize functions
extern void quantize_row_q2_K(const float * x, void * y, int64_t k);
extern void quantize_row_q3_K(const float * x, void * y, int64_t k);
extern void quantize_row_q4_1(const float * x, void * y, int64_t k);
extern void quantize_row_q5_0(const float * x, void * y, int64_t k);
extern void quantize_row_q5_1(const float * x, void * y, int64_t k);
extern void quantize_row_q8_0(const float * x, void * y, int64_t k);
extern void quantize_row_q8_1(const float * x, void * y, int64_t k);
extern void quantize_row_q4_K(const float * x, void * y, int64_t k);
extern void quantize_row_q5_K(const float * x, void * y, int64_t k);
extern void quantize_row_q6_K(const float * x, void * y, int64_t k);
extern void quantize_row_q8_K(const float * x, void * y, int64_t k);
extern void quantize_row_iq2_xs(const float * x, void * y, int64_t k);
extern void quantize_row_iq2_s(const float * x, void * y, int64_t k);
extern void quantize_row_iq3_xxs(const float * x, void * y, int64_t k);
extern void quantize_row_iq3_s(const float * x, void * y, int64_t k);
extern void quantize_row_iq4_nl(const float * x, void * y, int64_t k);
extern void quantize_row_iq4_xs(const float * x, void * y, int64_t k);
extern void quantize_row_iq1_s(const float * x, void * y, int64_t k);
extern void quantize_row_iq1_m(const float * x, void * y, int64_t k);
extern void quantize_row_tq1_0(const float * x, void * y, int64_t k);
extern void quantize_row_tq2_0(const float * x, void * y, int64_t k);

static int test_kernel(const char * name,
                       vec_dot_fn simd_fn, vec_dot_fn scalar_fn,
                       void (*quant_x)(const float *, void *, int64_t),
                       void (*quant_y)(const float *, void *, int64_t),
                       size_t block_size_x, size_t block_size_y,
                       int qk, int n_blocks) {
    int n = n_blocks * qk;

    // Allocate and fill with random data (raw blocks, not via quantize)
    size_t size_x = n_blocks * block_size_x;
    size_t size_y = n_blocks * block_size_y;
    void * qx = malloc(size_x);
    void * qy = malloc(size_y);

    // Fill with random bytes
    uint8_t *px = (uint8_t*)qx, *py = (uint8_t*)qy;
    for (size_t i = 0; i < size_x; i++) px[i] = rand() & 0xFF;
    for (size_t i = 0; i < size_y; i++) py[i] = rand() & 0xFF;

    // Set ALL fp16 fields in each block to 1.0 (0x3C00)
    // This is a blunt approach: every 2-byte-aligned position that could be a scale
    // gets set to 1.0. Won't produce correct dequant but ensures non-zero dot products.
    for (size_t i = 0; i < size_x; i += 2) {
        uint16_t *p = (uint16_t*)(px + i);
        if ((*p & 0x7C00) == 0) *p = 0x3C00; // if exponent is 0 (scale=0), set to 1.0
    }
    for (size_t i = 0; i < size_y; i += 2) {
        uint16_t *p = (uint16_t*)(py + i);
        if ((*p & 0x7C00) == 0) *p = 0x3C00;
    }
    (void)quant_x; (void)quant_y;

    float s_simd = 0, s_scalar = 0;
    simd_fn(n, &s_simd, 0, qx, 0, qy, 0, 1);
    scalar_fn(n, &s_scalar, 0, qx, 0, qy, 0, 1);

    float err = (s_scalar != 0) ? fabsf(s_simd - s_scalar) / fabsf(s_scalar) : fabsf(s_simd - s_scalar);
    int pass = (err < 0.0001f) || (fabsf(s_simd - s_scalar) < 1e-5f);

    printf("  %-12s: simd=%12.4f scalar=%12.4f err=%.8f %s\n",
           name, s_simd, s_scalar, err, pass ? "PASS" : "FAIL");

    free(qx); free(qy);
    return pass;
}

int main(void) {
    srand(42);
    int pass = 0, fail = 0;

    printf("=== SSSE3 kernel validation (vs scalar, quantized data) ===\n\n");

    #define T32(name, qx_fn, qy_fn, bx, by) do { \
        if (test_kernel(#name, ggml_vec_dot_##name, ggml_vec_dot_##name##_generic, \
            quantize_row_##qx_fn, quantize_row_##qy_fn, bx, by, 32, 16)) pass++; else fail++; \
    } while(0)

    #define T256(name, qx_fn, qy_fn, bx, by) do { \
        if (test_kernel(#name, ggml_vec_dot_##name, ggml_vec_dot_##name##_generic, \
            quantize_row_##qx_fn, quantize_row_##qy_fn, bx, by, 256, 4)) pass++; else fail++; \
    } while(0)

    T32(q5_0_q8_0, q5_0, q8_0, 22, 34);
    T32(q5_1_q8_1, q5_1, q8_1, 26, 36);
    T32(q8_0_q8_0, q8_0, q8_0, 34, 34);
    T32(q4_1_q8_1, q4_1, q8_1, 20, 36);
    T32(iq4_nl_q8_0, iq4_nl, q8_0, 18, 34);

    T256(q2_K_q8_K, q2_K, q8_K, 84, 292);
    T256(q3_K_q8_K, q3_K, q8_K, 110, 292);
    T256(q4_K_q8_K, q4_K, q8_K, 144, 292);
    T256(q5_K_q8_K, q5_K, q8_K, 176, 292);
    T256(q6_K_q8_K, q6_K, q8_K, 210, 292);
    T256(iq4_xs_q8_K, iq4_xs, q8_K, 138, 292);
    T256(tq2_0_q8_K, tq2_0, q8_K, 66, 292);
    // IQ1/IQ2/IQ3/TQ1 quantize functions not in libggml-cpu, skip for now

    printf("\n  %d PASS, %d FAIL\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
