// Shared FWHT (Fast Walsh-Hadamard Transform) core for TURBO_KV_4B.
// All turbo_kv_4b shaders include this for the subgroup-cooperative RHT.
//
// Requires: GL_KHR_shader_subgroup_shuffle (subgroupShuffleXor)
// Optional: GL_KHR_shader_subgroup_arithmetic (subgroupAdd, subgroupMax — for quantize)
//
// Wave64 (GCN): 64 threads × 2 values = 128 elements per block.
// Wave32 (RDNA): would need 32 threads × 4 values — different layout (not implemented).

// Lloyd-Max-Gaussian codebook for N(0,1), 16 centroids (Max 1960 tables)
const float TURBO_KV_CODEBOOK[16] = float[16](
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284,
     0.1284,  0.3881,  0.6568,  0.9423,  1.2562,  1.6180,  2.0690,  2.7326
);

const uint TURBO_KV_SEED = 0x12345678u;
const float TURBO_KV_RSQRT128 = 0.08838834764; // 1/sqrt(128)

// Knuth multiplicative hash for deterministic sign flips
float turbo_kv_sign(uint idx) {
    uint h = TURBO_KV_SEED ^ idx;
    h *= 2654435761u;
    return (h & 1u) != 0 ? 1.0 : -1.0;
}

// Subgroup-cooperative Walsh-Hadamard butterfly (7 stages for dim=128).
// Each thread holds val0 = data[tid] and val1 = data[tid+64].
// Stages 0-5: cross-thread via subgroupShuffleXor (strides 1-32).
// Stage 6: thread-local swap (stride 64).
void turbo_kv_fwht(inout float val0, inout float val1, uint tid) {
    [[unroll]] for (uint s = 1u; s <= 32u; s <<= 1) {
        float p0 = subgroupShuffleXor(val0, s);
        float p1 = subgroupShuffleXor(val1, s);
        if ((tid & s) == 0) { val0 += p0; val1 += p1; }
        else                { val0 = p0 - val0; val1 = p1 - val1; }
    }
    float t = val0;
    val0 = val0 + val1;
    val1 = t - val1;
}

// Inverse RHT: butterfly → normalize → sign_flip → scale
// Forward is: sign_flip → butterfly → normalize
// Inverse is: butterfly → normalize → sign_flip (sign only AFTER WHT, not before)
void turbo_kv_inverse_rht(inout float val0, inout float val1, uint tid, float norm) {
    turbo_kv_fwht(val0, val1, tid);
    float scale = norm * TURBO_KV_RSQRT128;
    val0 *= turbo_kv_sign(tid)      * scale;
    val1 *= turbo_kv_sign(tid + 64) * scale;
}

// Forward RHT: sign_flip → butterfly → normalize
void turbo_kv_forward_rht(inout float val0, inout float val1, uint tid) {
    val0 *= turbo_kv_sign(tid);
    val1 *= turbo_kv_sign(tid + 64);
    turbo_kv_fwht(val0, val1, tid);
    val0 *= TURBO_KV_RSQRT128;
    val1 *= TURBO_KV_RSQRT128;
}

// Nearest codebook entry. Codebook is sorted — unrolled linear scan.
uint turbo_kv_nearest(float x) {
    uint best = 0u;
    float best_dist = abs(x - TURBO_KV_CODEBOOK[0]);
    [[unroll]] for (uint c = 1u; c < 16u; c++) {
        float d = abs(x - TURBO_KV_CODEBOOK[c]);
        if (d < best_dist) { best = c; best_dist = d; }
    }
    return best;
}

// Unpack one 4-bit codebook index from a byte value.
float turbo_kv_unpack_byte(uint byte_val, uint elem, float rcp_inv_std) {
    uint idx = (elem & 1u) == 0 ? (byte_val & 0xFu) : (byte_val >> 4);
    return TURBO_KV_CODEBOOK[idx] * rcp_inv_std;
}
