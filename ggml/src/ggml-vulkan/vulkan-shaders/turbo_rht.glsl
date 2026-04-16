// turbo_rht.glsl — Parametric FWHT core for all TURBO_*B weight types
//
// Supports wave64 (GCN/CDNA) and wave32 (RDNA) via compile-time branching.
// Includes codebook tables at 2/3/4/5-bit and generic bit-unpacking.
//
// Requires: SUBGROUP_SIZE defined before including this file.
// Requires: GL_KHR_shader_subgroup_shuffle (subgroupShuffleXor)
// Optional: GL_KHR_shader_subgroup_arithmetic (subgroupAdd — for reductions)

#ifndef TURBO_RHT_GLSL
#define TURBO_RHT_GLSL

// ---- Published Lloyd-Max Gaussian codebooks (tq_codebook.c, Max 1960) ----

const float TURBO_CB_2BIT[4] = float[4](
    -1.5104, -0.4528, 0.4528, 1.5104
);

const float TURBO_CB_3BIT[8] = float[8](
    -2.1520, -1.3440, -0.7560, -0.2451,
     0.2451,  0.7560,  1.3440,  2.1520
);

const float TURBO_CB_4BIT[16] = float[16](
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284,
     0.1284,  0.3881,  0.6568,  0.9423,  1.2562,  1.6180,  2.0690,  2.7326
);

const float TURBO_CB_5BIT[32] = float[32](
    -1.9956, -1.7900, -1.6107, -1.4493, -1.3010, -1.1631, -1.0334, -0.9104,
    -0.7928, -0.6795, -0.5697, -0.4626, -0.3576, -0.2543, -0.1520, -0.0506,
     0.0506,  0.1520,  0.2543,  0.3576,  0.4626,  0.5697,  0.6795,  0.7928,
     0.9104,  1.0334,  1.1631,  1.3010,  1.4493,  1.6107,  1.7900,  1.9956
);

// ---- Constants ----

const uint  TURBO_SEED     = 0x12345678u;
const float TURBO_RSQRT128 = 0.08838834764831845; // 1/sqrt(128)

// ---- Deterministic sign from Knuth multiplicative hash ----

float turbo_sign(uint idx) {
    uint h = TURBO_SEED ^ idx;
    h *= 2654435761u;
    return ((h & 1u) != 0u) ? 1.0 : -1.0;
}

// ---- Generic bit-stream unpacking (matches CPU turbo_unpack_bits) ----
// Reads BITS-wide index at element position i from packed byte array.
// BITS must be defined as a specialization constant or compile define.

// Generic bit-stream unpacking.
// Caller provides two bytes straddling the target index.
// For use: read byte0 = buf[byte_idx], byte1 = buf[byte_idx+1],
// then call turbo_unpack_2bytes(byte0, byte1, bit_shift).
uint turbo_unpack_2bytes(uint byte0, uint byte1, uint bit_shift) {
    uint mask = (1u << BITS) - 1u;
    uint val = byte0 >> bit_shift;
    if (bit_shift + BITS > 8u) {
        val |= byte1 << (8u - bit_shift);
    }
    return val & mask;
}

// Convenience: compute byte index and shift from element index.
// Returns: x = byte_idx_relative, y = bit_shift
uvec2 turbo_bit_address(uint elem_idx) {
    uint bit_offset = elem_idx * BITS;
    return uvec2(bit_offset >> 3, bit_offset & 7u);
}

// ---- Codebook lookup (selects table based on BITS) ----

float turbo_codebook_lookup(uint idx) {
#if BITS == 2
    return TURBO_CB_2BIT[idx];
#elif BITS == 3
    return TURBO_CB_3BIT[idx];
#elif BITS == 4
    return TURBO_CB_4BIT[idx];
#elif BITS == 5
    return TURBO_CB_5BIT[idx];
#endif
}

// ================================================================
// Wave64 FWHT (64 threads × 2 values = 128 elements)
// ================================================================

#if SUBGROUP_SIZE >= 64

void turbo_fwht(inout float val0, inout float val1, uint tid) {
    [[unroll]] for (uint s = 1u; s <= 32u; s <<= 1) {
        float p0 = subgroupShuffleXor(val0, s);
        float p1 = subgroupShuffleXor(val1, s);
        if ((tid & s) == 0u) { val0 += p0; val1 += p1; }
        else                 { val0 = p0 - val0; val1 = p1 - val1; }
    }
    float t = val0;
    val0 = val0 + val1;
    val1 = t - val1;
}

void turbo_forward_rht(inout float val0, inout float val1, uint tid) {
    val0 *= turbo_sign(tid);
    val1 *= turbo_sign(tid + 64u);
    turbo_fwht(val0, val1, tid);
    val0 *= TURBO_RSQRT128;
    val1 *= TURBO_RSQRT128;
}

void turbo_inverse_rht(inout float val0, inout float val1, uint tid, float norm) {
    turbo_fwht(val0, val1, tid);
    float scale = norm * TURBO_RSQRT128;
    val0 *= turbo_sign(tid)       * scale;
    val1 *= turbo_sign(tid + 64u) * scale;
}

// ================================================================
// Wave32 FWHT (32 threads × 4 values = 128 elements)
// ================================================================

#else // SUBGROUP_SIZE < 64

void turbo_fwht(inout float v0, inout float v1, inout float v2, inout float v3, uint tid) {
    [[unroll]] for (uint s = 1u; s <= 16u; s <<= 1) {
        float o0 = subgroupShuffleXor(v0, s);
        float o1 = subgroupShuffleXor(v1, s);
        float o2 = subgroupShuffleXor(v2, s);
        float o3 = subgroupShuffleXor(v3, s);
        bool hi = (tid & s) != 0u;
        v0 = hi ? (o0 - v0) : (v0 + o0);
        v1 = hi ? (o1 - v1) : (v1 + o1);
        v2 = hi ? (o2 - v2) : (v2 + o2);
        v3 = hi ? (o3 - v3) : (v3 + o3);
    }
    float t0, t1;
    t0 = v0; t1 = v1; v0 = t0 + t1; v1 = t0 - t1;
    t0 = v2; t1 = v3; v2 = t0 + t1; v3 = t0 - t1;
    t0 = v0; t1 = v2; v0 = t0 + t1; v2 = t0 - t1;
    t0 = v1; t1 = v3; v1 = t0 + t1; v3 = t0 - t1;
}

void turbo_forward_rht(inout float v0, inout float v1, inout float v2, inout float v3, uint tid) {
    v0 *= turbo_sign(tid);
    v1 *= turbo_sign(tid + 32u);
    v2 *= turbo_sign(tid + 64u);
    v3 *= turbo_sign(tid + 96u);
    turbo_fwht(v0, v1, v2, v3, tid);
    v0 *= TURBO_RSQRT128;
    v1 *= TURBO_RSQRT128;
    v2 *= TURBO_RSQRT128;
    v3 *= TURBO_RSQRT128;
}

void turbo_inverse_rht(inout float v0, inout float v1, inout float v2, inout float v3, uint tid, float norm) {
    turbo_fwht(v0, v1, v2, v3, tid);
    float scale = norm * TURBO_RSQRT128;
    v0 *= turbo_sign(tid)       * scale;
    v1 *= turbo_sign(tid + 32u) * scale;
    v2 *= turbo_sign(tid + 64u) * scale;
    v3 *= turbo_sign(tid + 96u) * scale;
}

#endif // SUBGROUP_SIZE

// ---- FP16 helpers ----

float turbo_fp16_to_f32(uint16_t h) {
    return unpackHalf2x16(uint(h)).x;
}

uint16_t turbo_f32_to_fp16(float x) {
    return uint16_t(packHalf2x16(vec2(x, 0.0)) & 0xFFFFu);
}

#endif // TURBO_RHT_GLSL
