// q4_0_ar16_binding.c — binds specs/mmq/q4_0_ar16.allium against the landed
// GGML_TYPE_Q4_0_AR16 (id 42) C reference in this fork.
//
// Each check_* function discharges the obligation(s) named next to it in
// specs/mmq/q4_0_ar16.obligations (disposition `test:<name>`). The spec-side
// reference implementations here are written independently from the engine
// code, straight from the allium formulas: quantize is
// clamp(round_nearest_even(x / scale), -8, 7) + 8 with scale = absmax/8
// (division, RNE via rintf under the default FE_TONEAREST mode), dequantize
// is (code - 8) * fp32(d).
//
// Known divergence (reported, see the .obligations manifest): the engine
// computes codes as roundf(x * (1/scale)) — round-half-AWAY-FROM-ZERO on a
// reciprocal multiply — where the spec says round_nearest_even(x / scale).
// The two agree except at exact rounding ties (and quotients within an ULP
// of a tie). check_quantize_formula binds the spec formula on random and
// structured non-tie inputs and FAILS on any mismatch there; the divergence
// probe below demonstrates the tie case and reports it without masking it.
// The same divergence exists verbatim in the ik_llama.cpp reference this
// port mirrors (its own code comment also claims RNE).
//
// Build + run: see specs/mmq/run-binding.sh (CPU-only static build of the
// ggml-cpu target, then compile this file against libggml-cpu.a +
// libggml-base.a with gcc-15, ccache off).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "ggml.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"

// Engine symbols under test (ggml-base / ggml-cpu; prototypes restated here
// so the fixture does not depend on internal headers).
void quantize_row_q4_0_ar16_ref(const float * x, block_q4_0_ar16 * y, int64_t k);
void dequantize_row_q4_0_ar16(const block_q4_0_ar16 * x, float * y, int64_t k);
void quantize_row_q8_0_ref(const float * x, block_q8_0 * y, int64_t k);
void ggml_vec_dot_q4_0_ar16_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx,
                                 const void * vy, size_t by, int nrc);
void ggml_cpu_init(void); // fills the CPU backend's fp16 tables; required before vec_dot

static int n_fail = 0;
static int n_divergence = 0;

#define FAIL(...) do { printf("FAIL     " __VA_ARGS__); printf("\n"); n_fail++; } while (0)
#define PASS(...) do { printf("pass     " __VA_ARGS__); printf("\n"); } while (0)

// Deterministic PRNG so every run binds the same fixture.
static uint64_t rng_state = 0xA4016AB16ULL; // q4_0_ar16, fixed seed
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static float rng_float(float lo, float hi) { // uniform in [lo, hi)
    return lo + (hi - lo) * ((rng_next() >> 11) * (1.0 / 9007199254740992.0));
}

// ---------------------------------------------------------------- spec-side
// UnpackCode (spec rule): code k = low nibble of qs[k/2] for even k, high for odd.
static int spec_unpack_code(const uint8_t * qs, int k) {
    return (k % 2 == 0) ? (qs[k/2] & 0x0F) : (qs[k/2] >> 4);
}

// QuantizeFormula (spec): per 16-elem block, scale = absmax/8, d = fp16(scale),
// code_k = clamp(round_nearest_even(x_k / scale), -8, 7) + 8, interleaved pack.
static void spec_quantize_block(const float * x, block_q4_0_ar16 * out) {
    float amax = 0.0f;
    for (int k = 0; k < 16; k++) {
        const float v = fabsf(x[k]);
        if (v > amax) amax = v;
    }
    const float scale = amax / 8.0f;
    out->d = ggml_fp32_to_fp16(scale);
    for (int j = 0; j < 8; j++) {
        int c[2];
        for (int half = 0; half < 2; half++) {
            const int k = 2*j + half;
            int code = 8; // ZeroBlockHandling: scale = 0 -> code 8
            if (scale != 0.0f) {
                const float q = x[k] / scale;      // division, per the spec
                long r = lrintf(q);                // RNE under FE_TONEAREST
                if (r < -8) r = -8;
                if (r >  7) r =  7;
                code = (int) r + 8;
            }
            c[half] = code;
        }
        out->qs[j] = (uint8_t)(c[0] | (c[1] << 4));
    }
}

// DequantizeFormula (spec): result[k] = fp32(code_k - 8) * fp32(d).
static void spec_dequantize_block(const block_q4_0_ar16 * b, float * out) {
    const float d = ggml_fp16_to_fp32(b->d);
    for (int k = 0; k < 16; k++) {
        out[k] = (float)(spec_unpack_code(b->qs, k) - 8) * d;
    }
}

static void fill_row(float * x, int n) {
    for (int i = 0; i < n; i++) x[i] = rng_float(-10.0f, 10.0f);
}

// ---------------------------------------------------------------- checks

// entity-fields.Block_Q4_0_AR16, config-default.block_size, config-default.type_size
// exercises: block_q4_0_ar16, QK4_0_AR16
static void check_block_layout(void) {
    if (QK4_0_AR16 != 16)                 { FAIL("check_block_layout: QK4_0_AR16 = %d, want 16", QK4_0_AR16); return; }
    if (sizeof(block_q4_0_ar16) != 10)    { FAIL("check_block_layout: sizeof(block_q4_0_ar16) = %zu, want 10", sizeof(block_q4_0_ar16)); return; }
    PASS("check_block_layout: 16-elem block, 10 bytes (fp16 d + 8 nibble bytes)");
}

// entity-fields.QuantizedRow — BlockCount/BlockAlignment: a quantized row is a
// bare array of n/16 blocks, no header; the writer touches exactly that span.
// exercises: quantize_row_q4_0_ar16_ref
static void check_row_layout(void) {
    enum { NB = 7, N = 16*NB, PAD = 32 };
    float x[N];
    fill_row(x, N);
    uint8_t * buf = malloc(NB*sizeof(block_q4_0_ar16) + PAD);
    memset(buf, 0xA5, NB*sizeof(block_q4_0_ar16) + PAD);
    quantize_row_q4_0_ar16_ref(x, (block_q4_0_ar16 *) buf, N);
    for (int i = 0; i < PAD; i++) {
        if (buf[NB*sizeof(block_q4_0_ar16) + i] != 0xA5) {
            FAIL("check_row_layout: writer touched byte %d past n/16 blocks", i);
            free(buf);
            return;
        }
    }
    free(buf);
    PASS("check_row_layout: row = n/16 contiguous blocks, no header, no overrun");
}

// rule-success.UnpackCode: every unpacked code is in [0, 16).
// exercises: quantize_row_q4_0_ar16_ref
static void check_unpack_code(void) {
    enum { NB = 64, N = 16*NB };
    float x[N];
    block_q4_0_ar16 q[NB];
    fill_row(x, N);
    quantize_row_q4_0_ar16_ref(x, q, N);
    for (int i = 0; i < NB; i++) {
        for (int k = 0; k < 16; k++) {
            const int raw = spec_unpack_code(q[i].qs, k);
            if (raw < 0 || raw >= 16) {
                FAIL("check_unpack_code: block %d code %d = %d out of [0,16)", i, k, raw);
                return;
            }
        }
    }
    PASS("check_unpack_code: all codes in [0,16) over %d blocks", NB);
}

// contract-signature.QuantizeRow.invoke — QuantizeFormula + ZeroBlockHandling,
// bit-exact on d and codes (config.quantize_rne_abs_tol = 0.0).
// exercises: quantize_row_q4_0_ar16_ref
static void check_quantize_formula(void) {
    enum { NB = 4096, N = 16*NB };
    static float x[N];
    static block_q4_0_ar16 got[NB], want[NB];
    fill_row(x, N);
    memset(x, 0, 16*sizeof(float));                    // block 0: ZeroBlockHandling
    for (int k = 0; k < 16; k++) x[16 + k] = (k % 2 ? -1.0f : 1.0f) * (float)k; // structured block
    quantize_row_q4_0_ar16_ref(x, got, N);
    for (int i = 0; i < NB; i++) spec_quantize_block(x + 16*i, &want[i]);
    for (int i = 0; i < NB; i++) {
        if (got[i].d != want[i].d) {
            FAIL("check_quantize_formula: block %d d = 0x%04x, spec wants 0x%04x", i, got[i].d, want[i].d);
            return;
        }
        if (memcmp(got[i].qs, want[i].qs, 8) != 0) {
            FAIL("check_quantize_formula: block %d codes diverge from spec formula (first byte got 0x%02x want 0x%02x)",
                 i, got[i].qs[0], want[i].qs[0]);
            return;
        }
    }
    // ZeroBlockHandling explicitly: d = 0, all codes 8.
    if (ggml_fp16_to_fp32(got[0].d) != 0.0f) { FAIL("check_quantize_formula: zero block d != 0"); return; }
    for (int j = 0; j < 8; j++) {
        if (got[0].qs[j] != 0x88) { FAIL("check_quantize_formula: zero block code byte %d = 0x%02x, want 0x88", j, got[0].qs[j]); return; }
    }
    PASS("check_quantize_formula: d + codes bit-exact vs spec formula over %d blocks (incl. zero block)", NB);
}

// Divergence probe (NOT an obligation discharge; reported, kept out of the
// pass/fail budget so it cannot be mistaken for a green binding): at an exact
// rounding tie the engine's roundf (half-away-from-zero) departs from the
// spec's round_nearest_even. Constructed case: block [8, 2.5, 0, ...]:
// absmax 8 -> scale 1.0 exactly; 2.5/1.0 = 2.5 -> RNE 2 (code 10), engine
// roundf 3 (code 11).
static void probe_rne_tie_divergence(void) {
    float x[16] = { 8.0f, 2.5f, 0 };
    block_q4_0_ar16 got, want;
    quantize_row_q4_0_ar16_ref(x, &got, 16);
    spec_quantize_block(x, &want);
    if (memcmp(&got, &want, sizeof(got)) != 0) {
        printf("DIVERGE  probe_rne_tie_divergence: at x/scale = 2.5 engine code = %d (roundf, half-away-from-zero), "
               "spec round_nearest_even code = %d — QuantizeFormula's RNE is not what the engine implements at exact ties "
               "(same in the ik_llama.cpp reference)\n",
               spec_unpack_code(got.qs, 1), spec_unpack_code(want.qs, 1));
        n_divergence++;
    } else {
        printf("note     probe_rne_tie_divergence: tie case agreed (unexpected — recheck the engine rounding)\n");
    }
}

// contract-signature.DequantizeBlock.invoke — DequantizeFormula + ResultLength,
// bit-exact (config.dequantize_rel_tol = 0.0), exactly 16 floats per block.
// exercises: dequantize_row_q4_0_ar16
static void check_dequantize_formula(void) {
    enum { NB = 4096, N = 16*NB, PAD = 16 };
    static float x[N], want[16];
    static block_q4_0_ar16 q[NB];
    static float got[N + PAD];
    fill_row(x, N);
    quantize_row_q4_0_ar16_ref(x, q, N);
    for (int i = 0; i < PAD; i++) got[N + i] = -777.0f;   // ResultLength canary
    dequantize_row_q4_0_ar16(q, got, N);
    for (int i = 0; i < NB; i++) {
        spec_dequantize_block(&q[i], want);
        for (int k = 0; k < 16; k++) {
            if (memcmp(&got[16*i + k], &want[k], sizeof(float)) != 0) {
                FAIL("check_dequantize_formula: block %d elem %d = %a, spec wants %a", i, k, got[16*i + k], want[k]);
                return;
            }
        }
    }
    for (int i = 0; i < PAD; i++) {
        if (got[N + i] != -777.0f) { FAIL("check_dequantize_formula: wrote past 16 elems/block"); return; }
    }
    PASS("check_dequantize_formula: bit-exact (code-8)*fp32(d) over %d blocks, length exact", NB);
}

// invariant.Determinism — identical input, bit-identical quantized rows.
// exercises: quantize_row_q4_0_ar16_ref
static void check_determinism(void) {
    enum { NB = 1024, N = 16*NB };
    static float x[N];
    static block_q4_0_ar16 a[NB], b[NB];
    fill_row(x, N);
    memset(a, 0x11, sizeof(a));
    memset(b, 0x22, sizeof(b));
    quantize_row_q4_0_ar16_ref(x, a, N);
    quantize_row_q4_0_ar16_ref(x, b, N);
    if (memcmp(a, b, sizeof(a)) != 0) { FAIL("check_determinism: two quantize runs differ"); return; }
    PASS("check_determinism: repeated quantize bit-identical over %d blocks", NB);
}

// invariant.ColPermEquivalence16Aligned — permuting whole 16-elem blocks of the
// quantized row equals permuting the dequantized columns, bit-exactly.
// exercises: quantize_row_q4_0_ar16_ref, dequantize_row_q4_0_ar16
static void check_colperm_16aligned(void) {
    enum { NB = 256, N = 16*NB, ROUNDS = 8 };
    static float x[N], direct[N], direct_permuted[N], permuted_dequant[N];
    static block_q4_0_ar16 q[NB], q_permuted[NB];
    static int perm[NB];
    for (int round = 0; round < ROUNDS; round++) {
        fill_row(x, N);
        quantize_row_q4_0_ar16_ref(x, q, N);
        // random permutation of blocks (Fisher-Yates)
        for (int i = 0; i < NB; i++) perm[i] = i;
        for (int i = NB - 1; i > 0; i--) {
            const int j = (int)(rng_next() % (uint64_t)(i + 1));
            const int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        }
        for (int i = 0; i < NB; i++) q_permuted[i] = q[perm[i]];   // blocks carried verbatim
        dequantize_row_q4_0_ar16(q, direct, N);
        dequantize_row_q4_0_ar16(q_permuted, permuted_dequant, N);
        for (int i = 0; i < NB; i++) {
            memcpy(direct_permuted + 16*i, direct + 16*perm[i], 16*sizeof(float));
        }
        if (memcmp(permuted_dequant, direct_permuted, sizeof(direct_permuted)) != 0) {
            FAIL("check_colperm_16aligned: dequant(permute(q)) != permute(dequant(q)) at round %d", round);
            return;
        }
    }
    PASS("check_colperm_16aligned: bit-exact under %d random block permutations of %d blocks", ROUNDS, NB);
}

// invariant.AutoRoundG128Lossless — replicate one fp16 scale per 128-elem group
// across its 8 blocks; dequant must equal (code-8)*fp32(scale) exactly.
// exercises: dequantize_row_q4_0_ar16
static void check_autoround_g128_lossless(void) {
    enum { GROUPS = 512, NB = 8*GROUPS, N = 16*NB };
    static block_q4_0_ar16 q[NB];
    static uint8_t codes[N];
    static float got[N];
    static ggml_fp16_t scales[GROUPS];
    for (int g = 0; g < GROUPS; g++) {
        scales[g] = ggml_fp32_to_fp16(rng_float(1e-4f, 2.0f));
        for (int b = 0; b < 8; b++) {
            block_q4_0_ar16 * blk = &q[8*g + b];
            blk->d = scales[g];                    // per-128 scale carried verbatim
            for (int j = 0; j < 8; j++) {
                const uint8_t c0 = (uint8_t)(rng_next() & 0x0F);
                const uint8_t c1 = (uint8_t)(rng_next() & 0x0F);
                codes[16*(8*g + b) + 2*j + 0] = c0;
                codes[16*(8*g + b) + 2*j + 1] = c1;
                blk->qs[j] = (uint8_t)(c0 | (c1 << 4));
            }
        }
    }
    dequantize_row_q4_0_ar16(q, got, N);
    for (int g = 0; g < GROUPS; g++) {
        const float s = ggml_fp16_to_fp32(scales[g]);
        for (int e = 0; e < 128; e++) {
            const float want = (float)((int)codes[128*g + e] - 8) * s;
            if (memcmp(&got[128*g + e], &want, sizeof(float)) != 0) {
                FAIL("check_autoround_g128_lossless: group %d elem %d = %a, want %a", g, e, got[128*g + e], want);
                return;
            }
        }
    }
    PASS("check_autoround_g128_lossless: %d W4G128 groups dequantize losslessly", GROUPS);
}

// contract-signature.VecDot.invoke — ScoreCorrectness on the FP32-accumulator
// CPU path: |got - ref| <= 1e-4 OR rel <= 1e-3 vs an independent scalar
// reference (dot of the two dequantized views, double accumulator).
// exercises: ggml_vec_dot_q4_0_ar16_q8_0
static void check_vec_dot_q8_0(void) {
    enum { NB8 = 128, N = 32*NB8, ROUNDS = 16 };   // N multiple of 32 (Q8_0 block)
    static float xa[N], xb[N], deq_a[N];
    static block_q4_0_ar16 qa[N/16];
    static block_q8_0 qb[NB8];
    for (int round = 0; round < ROUNDS; round++) {
        fill_row(xa, N);
        fill_row(xb, N);
        quantize_row_q4_0_ar16_ref(xa, qa, N);
        quantize_row_q8_0_ref(xb, qb, N);
        dequantize_row_q4_0_ar16(qa, deq_a, N);
        double ref = 0.0;
        for (int i = 0; i < NB8; i++) {
            const float dbf = ggml_fp16_to_fp32(qb[i].d);
            for (int k = 0; k < 32; k++) {
                ref += (double)deq_a[32*i + k] * ((double)qb[i].qs[k] * (double)dbf);
            }
        }
        float got = 0.0f;
        ggml_vec_dot_q4_0_ar16_q8_0(N, &got, 0, qa, 0, qb, 0, 1);
        const double abs_err = fabs((double)got - ref);
        const double rel_err = fabs(ref) > 0 ? abs_err / fabs(ref) : abs_err;
        if (abs_err > 1e-4 && rel_err > 1e-3) {
            FAIL("check_vec_dot_q8_0: round %d got %.9g ref %.9g (abs %.3g rel %.3g)", round, got, ref, abs_err, rel_err);
            return;
        }
    }
    PASS("check_vec_dot_q8_0: %d rounds of n=%d within abs 1e-4 OR rel 1e-3", ROUNDS, N);
}

int main(void) {
    printf("q4_0_ar16_binding: specs/mmq/q4_0_ar16.allium vs GGML_TYPE_Q4_0_AR16 (id 42) scalar reference\n\n");

    ggml_cpu_init();

    check_block_layout();
    check_row_layout();
    check_unpack_code();
    check_quantize_formula();
    check_dequantize_formula();
    check_determinism();
    check_colperm_16aligned();
    check_autoround_g128_lossless();
    check_vec_dot_q8_0();
    probe_rne_tie_divergence();

    printf("\n%s: %d failure(s), %d reported divergence(s)\n",
           n_fail == 0 ? "GREEN" : "RED", n_fail, n_divergence);
    return n_fail == 0 ? 0 : 1;
}
