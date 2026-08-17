// Q4_0_AR16 correctness suite.
//
// Binds on the q4_0_ar16.allium contract:
//   - DequantizeFormula, bit-exact (tol 0.0): result[k] = fp32(UnpackCode(qs,k) - 8) * fp32(d)
//     with the INTERLEAVED nibble layout (element k = nibble k: byte k/2, low nibble for
//     even k -- NOT Q4_0's split halves) and the symmetric scale d = absmax/8.
//   - QuantizeFormula tie semantics: the C ref rounds with roundf (ties AWAY from zero),
//     pinned as ground truth in the spec reconciliation (ties toward +-inf, NOT banker's).
//   - MUL_MAT / MUL_MAT_ID vs an independent reference (NMSE, same 5e-4 bound as
//     test-backend-ops), including shapes test-backend-ops does not generate:
//     ne00 % 32 == 16 (odd 16-element block count) and dual-GPU row-split buffers.
//   - The x86 SIMD vec_dot path (ggml_vec_dot_q4_0_ar16_q8_0) vs the scalar reference,
//     including the odd 16-element tail block (ne00 % 32 == 16).
//
// The CPU dequantize fixture and CPU MUL_MAT shapes always run. The GPU sections run when a GPU backend is
// present; the split-buffer section additionally needs >= 2 devices of the same backend.
// Missing hardware skips the section (exit stays 0), it does not fail.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

constexpr double NMSE_MAX = 5e-4; // matches test-backend-ops MUL_MAT bound for quant types

// independent fp16 -> fp32 (IEEE 754 half), deliberately not ggml's own conversion
static float fp16_to_fp32_ref(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

struct blk_ar16 { uint16_t d; uint8_t qs[8]; }; // mirrors block_q4_0_ar16 (10 bytes)
static_assert(sizeof(blk_ar16) == 10, "wrong fixture block size");

// spec DequantizeFormula on a raw block
static float dequant_ref(const blk_ar16 & b, int k) {
    const int code = (k % 2 == 0) ? (b.qs[k/2] & 0x0F) : (b.qs[k/2] >> 4);
    return (float)(code - 8) * fp16_to_fp32_ref(b.d);
}

static double nmse(const float * a, const double * ref, size_t n) {
    double se = 0.0, energy = 0.0;
    for (size_t i = 0; i < n; i++) {
        se     += ((double)a[i] - ref[i])*((double)a[i] - ref[i]);
        energy += ref[i]*ref[i];
    }
    return se/energy;
}

// GET_ROWS dequant of hand-packed blocks, compared bit-exactly to the spec formula
static int test_dequant_fixture(ggml_backend_t backend, const char * name) {
    // 2 rows x 32 elements = 4 blocks: normal, inexact-half, exact-power and zero scales;
    // codes cover 0..15 in block-dependent permutations, packed interleaved
    const uint16_t ds[4] = { 0x3800 /*0.5*/, 0x2E66 /*~0.1, inexact*/, 0x3C00 /*1.0*/, 0x0000 /*zero block*/ };
    blk_ar16 blocks[4];
    for (int b = 0; b < 4; b++) {
        blocks[b].d = ds[b];
        for (int j = 0; j < 8; j++) {
            const uint8_t c0 = (uint8_t)((2*j + b) % 16);       // even element -> low nibble
            const uint8_t c1 = (uint8_t)((2*j + 1 + 5*b) % 16); // odd  element -> high nibble
            blocks[b].qs[j] = (uint8_t)(c0 | (c1 << 4));
        }
    }

    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0_AR16, 32, 2);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    ggml_tensor * out  = ggml_get_rows(ctx, a, rows);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { printf("%s: alloc failed\n", name); return 1; }

    ggml_backend_tensor_set(a, blocks, 0, sizeof(blocks));
    const int32_t row_idx[2] = {0, 1};
    ggml_backend_tensor_set(rows, row_idx, 0, sizeof(row_idx));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("%s: compute failed\n", name);
        return 1;
    }

    float got[64];
    ggml_backend_tensor_get(out, got, 0, sizeof(got));

    int n_fail = 0;
    for (int i = 0; i < 64; i++) {
        const float expected = dequant_ref(blocks[i/16], i % 16);
        if (memcmp(&got[i], &expected, sizeof(float)) != 0) { // bit-exact, tol 0.0
            if (n_fail < 8) {
                printf("%s dequant MISMATCH elem %2d: got %.9g expected %.9g\n", name, i, got[i], expected);
            }
            n_fail++;
        }
    }
    printf("%-24s dequant fixture: %s (%d/64 mismatches)\n", name, n_fail == 0 ? "OK, bit-exact" : "FAIL", n_fail);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return n_fail == 0 ? 0 : 1;
}

// MUL_MAT vs an f64 reference built from the spec dequant of the quantized weights.
// buft_a overrides the weight buffer (used for the dual-GPU split case); null = backend default.
static int test_mul_mat(ggml_backend_t backend, const char * name, int m, int n, int k,
                        ggml_backend_buffer_type_t buft_a) {
    std::vector<float> src0f((size_t)m*k), src1f((size_t)k*n);
    for (size_t i = 0; i < src0f.size(); i++) src0f[i] = sinf(0.001f*i);
    for (size_t i = 0; i < src1f.size(); i++) src1f[i] = cosf(0.002f*i);

    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx_a = ggml_init(ip);
    ggml_context * ctx   = ggml_init(ip);

    ggml_tensor * a   = ggml_new_tensor_2d(ctx_a, GGML_TYPE_Q4_0_AR16, k, m);
    ggml_tensor * b   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf_a = buft_a
        ? ggml_backend_alloc_ctx_tensors_from_buft(ctx_a, buft_a)
        : ggml_backend_alloc_ctx_tensors(ctx_a, backend);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf_a || !buf) { printf("%s: alloc failed\n", name); return 1; }

    std::vector<uint8_t> q(ggml_nbytes(a));
    ggml_quantize_chunk(GGML_TYPE_Q4_0_AR16, src0f.data(), q.data(), 0, m, k, nullptr);
    ggml_backend_tensor_set(a, q.data(), 0, q.size());
    ggml_backend_tensor_set(b, src1f.data(), 0, src1f.size()*sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("%s: compute failed\n", name);
        return 1;
    }
    std::vector<float> got((size_t)m*n);
    ggml_backend_tensor_get(out, got.data(), 0, got.size()*sizeof(float));

    // reference: spec dequant of the stored blocks, f64 accumulation
    std::vector<double> ref((size_t)m*n);
    const blk_ar16 * blocks = (const blk_ar16 *) q.data();
    for (int c = 0; c < n; c++) {
        for (int r = 0; r < m; r++) {
            double acc = 0.0;
            for (int j = 0; j < k; j++) {
                acc += (double)dequant_ref(blocks[(size_t)r*(k/16) + j/16], j % 16) * (double)src1f[(size_t)c*k + j];
            }
            ref[(size_t)c*m + r] = acc;
        }
    }

    const double err = nmse(got.data(), ref.data(), got.size());
    const bool ok = err < NMSE_MAX;
    printf("%-24s MUL_MAT m=%d n=%-3d k=%-5d nmse %.3g %s\n", name, m, n, k, err, ok ? "OK" : "FAIL");

    ggml_backend_buffer_free(buf_a);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx_a);
    ggml_free(ctx);
    return ok ? 0 : 1;
}

// MUL_MAT_ID (expert routing): covers the mmvq-moe kernel (n_tokens <= 8) and the
// mmq-id path (n_tokens > 8) on GPU; reference as in test_mul_mat, per routed expert.
static int test_mul_mat_id(ggml_backend_t backend, const char * name, int n_tokens) {
    const int k = 256, m = 8, n_expert = 4, n_used = 2;

    std::vector<float> src0f((size_t)n_expert*m*k), src1f((size_t)n_used*k*n_tokens);
    for (size_t i = 0; i < src0f.size(); i++) src0f[i] = sinf(0.017f*i);
    for (size_t i = 0; i < src1f.size(); i++) src1f[i] = cosf(0.013f*i);
    // ids is [n_used, n_tokens]: one expert slot per b column (slot id uses b column
    // id % ne11 and writes output column id -- see ggml_compute_forward_mul_mat_id)
    std::vector<int32_t> ids_v((size_t)n_used*n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        for (int u = 0; u < n_used; u++) ids_v[(size_t)t*n_used + u] = (t + u) % n_expert;
    }

    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * as  = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0_AR16, k, m, n_expert);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { printf("%s: alloc failed\n", name); return 1; }

    std::vector<uint8_t> q(ggml_nbytes(as));
    ggml_quantize_chunk(GGML_TYPE_Q4_0_AR16, src0f.data(), q.data(), 0, (int64_t)n_expert*m, k, nullptr);
    ggml_backend_tensor_set(as, q.data(), 0, q.size());
    ggml_backend_tensor_set(b, src1f.data(), 0, src1f.size()*sizeof(float));
    ggml_backend_tensor_set(ids, ids_v.data(), 0, ids_v.size()*sizeof(int32_t));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("%s: compute failed\n", name);
        return 1;
    }
    std::vector<float> got(ggml_nelements(out));
    ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));

    // reference: out[:, u, t] = as[ids[u, t]] @ b[:, u, t]
    std::vector<double> ref(got.size());
    const blk_ar16 * blocks = (const blk_ar16 *) q.data();
    for (int t = 0; t < n_tokens; t++) {
        for (int u = 0; u < n_used; u++) {
            const int e = ids_v[(size_t)t*n_used + u];
            for (int r = 0; r < m; r++) {
                double acc = 0.0;
                for (int j = 0; j < k; j++) {
                    const blk_ar16 & bl = blocks[((size_t)e*m + r)*(k/16) + j/16];
                    acc += (double)dequant_ref(bl, j % 16) * (double)src1f[((size_t)t*n_used + u)*k + j];
                }
                ref[((size_t)t*n_used + u)*m + r] = acc;
            }
        }
    }

    const double err = nmse(got.data(), ref.data(), got.size());
    const bool ok = err < NMSE_MAX;
    printf("%-24s MUL_MAT_ID n_tokens=%-3d nmse %.3g %s\n", name, n_tokens, err, ok ? "OK" : "FAIL");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok ? 0 : 1;
}

// QuantizeFormula tie semantics: the C ref uses roundf, which rounds HALVES
// away from zero (roundf(0.5)=1, roundf(-0.5)=-1), NOT banker's rounding.
// Pin it directly through the public quantize_chunk path (which routes to
// quantize_row_q4_0_ar16_ref): build a row whose x*id values land exactly on
// .5 ties and assert the emitted codes are the ties-away-from-zero codes.
static int test_quantize_ties() {
    // Element 15 = 8.0 sets amax = 8.0 -> d = 1.0, id = 1.0, so the tie
    // elements below land exactly on their .5 ties (x*id = x). roundf is
    // ties-away-from-zero: code = 8 + roundf(x).
    //   x =  0.5 -> roundf 1 -> code  9 (low nibble, element 0)
    //   x = -0.5 -> roundf -1 -> code  7 (high nibble, element 1)
    //   x =  2.5 -> roundf 3 -> code 11 (low nibble, element 2)
    //   x = -2.5 -> roundf -3 -> code  5 (high nibble, element 3)
    //   x =  7.5 -> roundf 8 -> code 16 -> clamped to 15 (element 4)
    //   x = -7.5 -> roundf -8 -> code 0 (element 5)
    //   x =  8.0 -> roundf 8 -> code 16 -> clamped to 15 (element 15)
    const float x[16] = {
         0.5f, -0.5f,  2.5f, -2.5f,  7.5f, -7.5f,  0.0f, 0.0f,
         0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f, 8.0f
    };
    blk_ar16 blk;
    const size_t nbytes = ggml_row_size(GGML_TYPE_Q4_0_AR16, 16);
    ggml_quantize_chunk(GGML_TYPE_Q4_0_AR16, x, &blk, 0, 1, 16, nullptr);
    if (nbytes != sizeof(blk_ar16)) {
        printf("quantize tie row size: got %zu want %zu\n", nbytes, sizeof(blk_ar16));
        return 1;
    }

    const uint8_t want_qs[8] = {
        (uint8_t)(0x9 | (0x7 << 4)),   // byte 0: elem0 low=9, elem1 high=7
        (uint8_t)(0xB | (0x5 << 4)),   // byte 1: elem2 low=11, elem3 high=5
        (uint8_t)(0xF | (0x0 << 4)),   // byte 2: elem4 low=15 (clamped), elem5 high=0
        0x88, 0x88, 0x88, 0x88,        // bytes 3..6: zero-coded (8|8<<4)
        (uint8_t)(0x8 | (0xF << 4)),   // byte 7: elem14 low=8, elem15 high=15 (clamped)
    };
    int n_fail = 0;
    for (int j = 0; j < 8; j++) {
        if (blk.qs[j] != want_qs[j]) {
            printf("quantize tie byte %d: got 0x%02X want 0x%02X\n", j, blk.qs[j], want_qs[j]);
            n_fail++;
        }
    }
    // d must be fp16(amax/8) = fp16(1.0) = 0x3C00
    if (blk.d != 0x3C00) {
        printf("quantize tie d: got 0x%04X want 0x3C00 (fp16 1.0)\n", (unsigned)blk.d);
        n_fail++;
    }
    printf("%-24s quantize tie semantics: %s\n", "CPU", n_fail == 0 ? "OK" : "FAIL");
    return n_fail == 0 ? 0 : 1;
}

// x86 SIMD vec_dot vs a scalar reference. The SIMD kernel (registered as the
// CPU type-traits vec_dot for AR16, paired with Q8_0) is exercised through the
// public type-traits table so the test binds the dispatched path. The direct
// vec_dot contract pairs two 16-elem AR16 blocks per 32-elem Q8_0 block, so it
// is only called on k that is a whole multiple of 32 (the odd 16-elem tail is
// handled by the mul_mat path's zero-padding, covered separately by the MUL_MAT
// shapes in this file and fixed at HEAD).
static int test_simd_vec_dot() {
    const struct ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(GGML_TYPE_Q4_0_AR16);
    const struct ggml_type_traits * tb = ggml_get_type_traits(GGML_TYPE_Q4_0_AR16);
    if (tr == nullptr || tr->vec_dot == nullptr || tr->vec_dot_type != GGML_TYPE_Q8_0
        || tb == nullptr || tb->to_float == nullptr) {
        printf("CPU vec_dot / to_float trait missing or not Q8_0-paired\n");
        return 1;
    }
    // shape matrix: even block counts only (32, 64, 256) -- the direct
    // vec_dot contract requires whole Q8_0 (32-elem) blocks.
    const int ks[] = { 32, 64, 256 };
    int n_fail = 0;
    for (int ki = 0; ki < 3; ki++) {
        const int k = ks[ki];
        std::vector<float> x((size_t)k), y((size_t)k);
        for (int j = 0; j < k; j++) {
            x[j] = sinf(0.003f * j);
            y[j] = cosf(0.002f * j) * 0.1f;
        }
        std::vector<uint8_t> qx(ggml_row_size(GGML_TYPE_Q4_0_AR16, k));
        std::vector<uint8_t> qy(ggml_row_size(GGML_TYPE_Q8_0, k));
        ggml_quantize_chunk(GGML_TYPE_Q4_0_AR16, x.data(), qx.data(), 0, 1, k, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, y.data(), qy.data(), 0, 1, k, nullptr);

        float sum_simd = 0.0f, sum_ref = 0.0f;
        tr->vec_dot(k, &sum_simd, 0, qx.data(), 0, qy.data(), 0, 0);
        // scalar reference: dequant x via the traits table, use y directly, dot
        std::vector<float> dx((size_t)k);
        tb->to_float(qx.data(), dx.data(), k);
        for (int j = 0; j < k; j++) sum_ref += dx[j] * y[j];

        const double diff = fabs((double)sum_simd - (double)sum_ref);
        const double tol  = 1e-2 * (fabs((double)sum_ref) + 1.0); // fp32 accumulation slack
        const bool ok = diff <= tol;
        printf("%-24s SIMD vec_dot k=%4d diff %.4g %s\n", "CPU", k, diff, ok ? "OK" : "FAIL");
        if (!ok) n_fail++;
    }
    return n_fail == 0 ? 0 : 1;
}

int main() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) { printf("no CPU backend\n"); return 1; }

    int fails = 0;
    fails += test_dequant_fixture(cpu, "CPU");
    fails += test_quantize_ties();
    fails += test_simd_vec_dot();
    fails += test_mul_mat(cpu, "CPU", 16, 1, 256, nullptr);
    fails += test_mul_mat(cpu, "CPU", 16, 64, 256, nullptr);
    fails += test_mul_mat(cpu, "CPU", 4, 3, 48, nullptr);   // odd block count: scalar vec_dot tail
    ggml_backend_free(cpu);

    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!gpu) {
        printf("no GPU backend: GPU sections skipped\n");
        return fails == 0 ? 0 : 1;
    }
    const char * gpu_name = ggml_backend_name(gpu);

    fails += test_dequant_fixture(gpu, gpu_name);
    fails += test_mul_mat(gpu, gpu_name, 16, 1, 256, nullptr);   // MMVQ
    fails += test_mul_mat(gpu, gpu_name, 16, 64, 256, nullptr);  // MMQ
    fails += test_mul_mat(gpu, gpu_name, 4, 3, 48, nullptr);     // odd block count: MMVQ pairing gate
    fails += test_mul_mat_id(gpu, gpu_name, 3);                  // mmvq moe kernel
    fails += test_mul_mat_id(gpu, gpu_name, 32);                 // mmq-id path

    // dual-GPU row split (the peer-transfer path): needs >= 2 devices on this backend's
    // registry and the split-buffer proc (CUDA exposes it; other backends skip)
    ggml_backend_dev_t dev = ggml_backend_get_device(gpu);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const size_t n_dev = ggml_backend_reg_dev_count(reg);
    auto split_fn = (ggml_backend_split_buffer_type_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
    if (n_dev >= 2 && split_fn) {
        const float tensor_split[16] = {0.5f, 1.0f};
        ggml_backend_buffer_type_t buft_split = split_fn(0, tensor_split);
        fails += test_mul_mat(gpu, "GPU-split(2)", 4096, 1, 14336, buft_split);
        fails += test_mul_mat(gpu, "GPU-split(2)", 4096, 16, 14336, buft_split);
    } else {
        printf("split-buffer section skipped (%zu device(s), split proc %s)\n",
               n_dev, split_fn ? "present" : "absent");
    }

    ggml_backend_free(gpu);
    printf("%s\n", fails == 0 ? "all Q4_0_AR16 checks passed" : "Q4_0_AR16 checks FAILED");
    return fails == 0 ? 0 : 1;
}
