/*
 * ggml-turbo-e8p.c — E8P lattice vector quantizer for TURBO_4B weights
 *
 * Implements the E8P codebook from QuIP# (arXiv 2402.04396):
 *   D8-hat lattice (all-half-integer 8D vectors, coordinate sum even)
 *   256 entries: 227 with norm² ≤ 10, 29 padding with norm² = 12
 *   RVQ at 4 bpw: two 16-bit E8P codes per group of 8 elements
 *
 * The codebook is generated programmatically at first use, not stored
 * as a compile-time constant table.
 */

#include "ggml-turbo-kv.h"
#include "ggml.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ================================================================
 * E8P Codebook (256 entries × 8 absolute coordinate values)
 *
 * Generated from D8-hat: all-half-integer 8-vectors with even
 * coordinate sum. Coordinates are from {0.5, 1.5, 2.5, 3.5}.
 * First 227 entries: norm² ≤ 10. Last 29: norm² = 12.
 * ================================================================ */

/* Absolute coordinate values for each codebook entry */
static float e8p_codebook_abs[256][8];
/* Squared norms of codebook entries */
static float e8p_codebook_norm2[256];
/* Whether the codebook has been initialized */
static int e8p_initialized = 0;

/* Compare function for sorting codebook entries by norm then lexicographic */
typedef struct { float abs[8]; float norm2; } e8p_entry_t;

static int e8p_entry_cmp(const void * a, const void * b) {
    const e8p_entry_t * ea = (const e8p_entry_t *)a;
    const e8p_entry_t * eb = (const e8p_entry_t *)b;
    /* Sort by norm first */
    if (ea->norm2 < eb->norm2) return -1;
    if (ea->norm2 > eb->norm2) return  1;
    /* Then lexicographic by coordinates */
    for (int i = 0; i < 8; i++) {
        if (ea->abs[i] < eb->abs[i]) return -1;
        if (ea->abs[i] > eb->abs[i]) return  1;
    }
    return 0;
}

static void e8p_init_codebook(void) {
    if (e8p_initialized) return;

    /* Half-integer coordinate values */
    static const float coords[4] = { 0.5f, 1.5f, 2.5f, 3.5f };

    /* Enumerate all D8-hat vectors and collect by norm */
    e8p_entry_t entries[4096]; /* upper bound */
    int n_entries = 0;

    /* 4^8 = 65536 combinations of absolute coordinates */
    int idx[8];
    for (idx[0] = 0; idx[0] < 4; idx[0]++)
    for (idx[1] = 0; idx[1] < 4; idx[1]++)
    for (idx[2] = 0; idx[2] < 4; idx[2]++)
    for (idx[3] = 0; idx[3] < 4; idx[3]++)
    for (idx[4] = 0; idx[4] < 4; idx[4]++)
    for (idx[5] = 0; idx[5] < 4; idx[5]++)
    for (idx[6] = 0; idx[6] < 4; idx[6]++)
    for (idx[7] = 0; idx[7] < 4; idx[7]++) {
        float abs_vals[8], norm2 = 0;
        float coord_sum = 0;
        for (int i = 0; i < 8; i++) {
            abs_vals[i] = coords[idx[i]];
            norm2 += abs_vals[i] * abs_vals[i];
            coord_sum += abs_vals[i];
        }

        /* D8-hat constraint: coordinate sum must be even.
         * Since coordinates are half-integers (n+0.5), their sum is
         * sum(n_i) + 4.0. For the sum to be even, sum(n_i) must be even.
         * Equivalently: the number of odd idx values must be even. */
        int n_odd = 0;
        for (int i = 0; i < 8; i++) {
            if (idx[i] % 2 != 0) n_odd++;
        }
        if (n_odd % 2 != 0) continue;

        /* Keep entries with norm² ≤ 12 (norm² ≤ 10 for main, ≤ 12 for padding) */
        if (norm2 > 12.0f + 0.01f) continue;

        GGML_ASSERT(n_entries < 4096);
        memcpy(entries[n_entries].abs, abs_vals, 8 * sizeof(float));
        entries[n_entries].norm2 = norm2;
        n_entries++;
    }

    /* Sort by norm, then lexicographic */
    qsort(entries, n_entries, sizeof(e8p_entry_t), e8p_entry_cmp);

    /* Take first 256 entries: should be 227 with norm² ≤ 10, plus some norm² = 12 */
    GGML_ASSERT(n_entries >= 256);

    for (int i = 0; i < 256; i++) {
        memcpy(e8p_codebook_abs[i], entries[i].abs, 8 * sizeof(float));
        e8p_codebook_norm2[i] = entries[i].norm2;
    }

    e8p_initialized = 1;
}

/* ================================================================
 * E8P 16-bit encode/decode (2 bpw)
 *
 * 16-bit code layout:
 *   bits 15-8: codebook index (0-255)
 *   bits  7-0: sign mask (8 bits, one per coordinate)
 *              bit 0 has parity XOR with overall sign parity
 *
 * The ±0.25 shift from QuIP# is applied based on sign parity.
 * ================================================================ */

void e8p_decode_16bit(uint16_t code, float * out) {
    if (!e8p_initialized) e8p_init_codebook();

    uint8_t abs_idx   = code >> 8;
    uint8_t sign_mask = code & 0xFF;

    /* Parity of sign_mask determines ±0.25 shift */
    int parity = __builtin_popcount(sign_mask) & 1;
    float shift = parity ? -0.25f : 0.25f;

    /* Remove parity from bit 0 to get actual signs */
    uint8_t actual_signs = sign_mask ^ parity;

    for (int i = 0; i < 8; i++) {
        float abs_v = e8p_codebook_abs[abs_idx][i];
        float sign  = ((actual_signs >> i) & 1) ? -1.0f : 1.0f;
        out[i] = abs_v * sign + shift;
    }
}

uint16_t e8p_encode_16bit(const float * x) {
    if (!e8p_initialized) e8p_init_codebook();

    uint16_t best_code = 0;
    float best_err = 1e30f;

    for (int parity_flag = 0; parity_flag < 2; parity_flag++) {
        float offset = parity_flag ? 0.25f : -0.25f;

        /* Apply offset and extract signs */
        float abs_x[8];
        int signs[8];
        int n_neg = 0;
        for (int i = 0; i < 8; i++) {
            float v = x[i] + offset;
            signs[i] = (v < 0.0f) ? 1 : 0;
            abs_x[i] = fabsf(v);
            n_neg += signs[i];
        }

        /* D8-hat requires even number of negative coordinates.
         * If odd, flip the sign of the coordinate closest to zero. */
        if (n_neg % 2 != 0) {
            int min_idx = 0;
            float min_val = abs_x[0];
            for (int i = 1; i < 8; i++) {
                if (abs_x[i] < min_val) { min_val = abs_x[i]; min_idx = i; }
            }
            signs[min_idx] ^= 1;
        }

        /* Find nearest codebook entry by maximum inner product:
         * ||abs_x - cb||² = ||abs_x||² - 2<abs_x, cb> + ||cb||²
         * Minimizing this is equivalent to maximizing 2<abs_x, cb> - ||cb||² */
        int best_idx = 0;
        float best_score = -1e30f;
        for (int idx = 0; idx < 256; idx++) {
            float score = -e8p_codebook_norm2[idx];
            for (int i = 0; i < 8; i++) {
                score += 2.0f * abs_x[i] * e8p_codebook_abs[idx][i];
            }
            if (score > best_score) { best_score = score; best_idx = idx; }
        }

        /* Compute reconstruction error */
        float err = 0;
        for (int i = 0; i < 8; i++) {
            float q = (signs[i] ? -e8p_codebook_abs[best_idx][i]
                                :  e8p_codebook_abs[best_idx][i]) - offset;
            float d = x[i] - q;
            err += d * d;
        }

        if (err < best_err) {
            best_err = err;
            uint8_t sign_mask = 0;
            for (int i = 0; i < 8; i++) {
                if (signs[i]) sign_mask |= (1 << i);
            }
            /* XOR bit 0 with parity flag */
            if (parity_flag) sign_mask ^= 1;
            best_code = ((uint16_t)best_idx << 8) | sign_mask;
        }
    }
    return best_code;
}

/* ================================================================
 * E8P RVQ at 4 bpw: two 16-bit E8P codes per group of 8
 *
 * Encode: quantize x → code1, compute residual, quantize residual → code2
 * Decode: decode code1, decode code2, result = coarse + fine / OPT_RESID_SCALE
 * ================================================================ */

#define E8P_OPT_RESID_SCALE 3.45f

void e8p_decode_rvq4bit(uint32_t code, float * out) {
    uint16_t code1 = (uint16_t)(code >> 16);
    uint16_t code2 = (uint16_t)(code & 0xFFFF);

    float coarse[8], fine[8];
    e8p_decode_16bit(code1, coarse);
    e8p_decode_16bit(code2, fine);

    for (int i = 0; i < 8; i++) {
        out[i] = coarse[i] + fine[i] / E8P_OPT_RESID_SCALE;
    }
}

uint32_t e8p_encode_rvq4bit(const float * x) {
    uint16_t code1 = e8p_encode_16bit(x);

    float coarse[8];
    e8p_decode_16bit(code1, coarse);

    /* Residual scaled up for better quantization */
    float residual[8];
    for (int i = 0; i < 8; i++) {
        residual[i] = (x[i] - coarse[i]) * E8P_OPT_RESID_SCALE;
    }

    uint16_t code2 = e8p_encode_16bit(residual);
    return ((uint32_t)code1 << 16) | (uint32_t)code2;
}

/* ================================================================
 * E8P + imatrix weighted encode: minimize weighted MSE
 *
 * For each group of 8 elements, minimizes:
 *   sum_i( weight[i] * (x[i] - decode(encode(x))[i])² )
 * ================================================================ */

uint32_t e8p_encode_rvq4bit_weighted(const float * x, const float * weights) {
    if (!weights) return e8p_encode_rvq4bit(x);

    /* For the coarse code, try both parities and all 256 codebook entries
     * with weighted distance metric */
    if (!e8p_initialized) e8p_init_codebook();

    uint16_t best_code1 = 0;
    float best_werr = 1e30f;

    for (int parity_flag = 0; parity_flag < 2; parity_flag++) {
        float offset = parity_flag ? 0.25f : -0.25f;

        float abs_x[8];
        int signs[8];
        int n_neg = 0;
        for (int i = 0; i < 8; i++) {
            float v = x[i] + offset;
            signs[i] = (v < 0.0f) ? 1 : 0;
            abs_x[i] = fabsf(v);
            n_neg += signs[i];
        }
        if (n_neg % 2 != 0) {
            /* Flip sign of element with lowest importance-weighted impact */
            int min_idx = 0;
            float min_cost = weights[0] * abs_x[0] * abs_x[0];
            for (int i = 1; i < 8; i++) {
                float cost = weights[i] * abs_x[i] * abs_x[i];
                if (cost < min_cost) { min_cost = cost; min_idx = i; }
            }
            signs[min_idx] ^= 1;
        }

        for (int idx = 0; idx < 256; idx++) {
            float werr = 0;
            for (int i = 0; i < 8; i++) {
                float q = (signs[i] ? -e8p_codebook_abs[idx][i]
                                    :  e8p_codebook_abs[idx][i]) - offset;
                float d = x[i] - q;
                werr += weights[i] * d * d;
            }
            if (werr < best_werr) {
                best_werr = werr;
                uint8_t sign_mask = 0;
                for (int i = 0; i < 8; i++) {
                    if (signs[i]) sign_mask |= (1 << i);
                }
                if (parity_flag) sign_mask ^= 1;
                best_code1 = ((uint16_t)idx << 8) | sign_mask;
            }
        }
    }

    /* Decode coarse and encode residual (unweighted for simplicity) */
    float coarse[8];
    e8p_decode_16bit(best_code1, coarse);
    float residual[8];
    for (int i = 0; i < 8; i++) {
        residual[i] = (x[i] - coarse[i]) * E8P_OPT_RESID_SCALE;
    }
    uint16_t code2 = e8p_encode_16bit(residual);

    return ((uint32_t)best_code1 << 16) | (uint32_t)code2;
}
