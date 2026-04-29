// Shared FWHT (Fast Walsh-Hadamard Transform) core for TURBO_KV_4B.
// All turbo_kv_4b shaders include this for the subgroup-cooperative RHT.
//
// Requires: GL_KHR_shader_subgroup_shuffle (subgroupShuffleXor)
// Optional: GL_KHR_shader_subgroup_arithmetic (subgroupAdd, subgroupMax — for quantize)
//
// Layout: 128-thread workgroup, one value per lane (val = data[tid]).
// Works for any subgroup size in {32, 64} — Turing/Ampere/Ada/RDNA all sg=32,
// GCN/Vega/MI200 sg=64. SUBGROUP_SIZE is a spec constant set by the dispatcher
// to device->subgroup_size; the dispatcher also pins requiredSubgroupSize to
// the same value via VK_EXT_subgroup_size_control so the unrolled bounds match
// the runtime subgroup width.
//
// The 7-stage butterfly (log2(128)) splits naturally:
//   - Stages with stride < SUBGROUP_SIZE: subgroupShuffleXor (no barrier).
//   - Stages with stride >= SUBGROUP_SIZE: shared memory + barrier.
// On sg=32 that's 5 cheap stages + 2 LDS stages. On sg=64: 6 cheap + 1 LDS.
// Pattern matches cumsum.comp:15-17 / mul_mat_vec_base.glsl:154-200 in this
// repo for portable subgroup-size-agnostic reductions.

layout (constant_id = 0) const uint SUBGROUP_SIZE = 32;

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

// Shared memory for the cross-subgroup butterfly stages and for the
// per-block scalar reductions (sum_sq / max_abs). 128 floats covers both:
// the butterfly uses lds[0..128), the cross-subgroup reduce uses lds[0..4)
// (max 4 partials at sg=32). They're not used simultaneously.
shared float turbo_kv_lds[128];

// Walsh-Hadamard butterfly, 7 stages for dim=128, one value per lane.
void turbo_kv_fwht(inout float val, uint tid) {
    // Subgroup-internal stages: shuffle, no barrier.
    [[unroll]] for (uint s = 1u; s < SUBGROUP_SIZE; s <<= 1) {
        float p = subgroupShuffleXor(val, s);
        val = ((tid & s) == 0u) ? (val + p) : (p - val);
    }
    // Cross-subgroup stages: shared memory.
    [[unroll]] for (uint s = SUBGROUP_SIZE; s < 128u; s <<= 1) {
        barrier();
        turbo_kv_lds[tid] = val;
        barrier();
        float p = turbo_kv_lds[tid ^ s];
        val = ((tid & s) == 0u) ? (val + p) : (p - val);
    }
}

// Inverse RHT: butterfly → normalize → sign_flip → scale
// Forward is: sign_flip → butterfly → normalize
// Inverse is: butterfly → normalize → sign_flip (sign only AFTER WHT, not before)
void turbo_kv_inverse_rht(inout float val, uint tid, float norm) {
    turbo_kv_fwht(val, tid);
    float scale = norm * TURBO_KV_RSQRT128;
    val *= turbo_kv_sign(tid) * scale;
}

// Forward RHT: sign_flip → butterfly → normalize
void turbo_kv_forward_rht(inout float val, uint tid) {
    val *= turbo_kv_sign(tid);
    turbo_kv_fwht(val, tid);
    val *= TURBO_KV_RSQRT128;
}

// Cross-subgroup sum reduce over the 128-thread workgroup.
// Uses the same shared LDS as the butterfly; safe because callers guard
// with their own pre-barrier when interleaving.
shared float turbo_kv_sh_partial[4]; // max 128/32 = 4 subgroups

float turbo_kv_workgroup_sum(float val) {
    float partial = subgroupAdd(val);
    if (subgroupElect()) turbo_kv_sh_partial[gl_SubgroupID] = partial;
    barrier();
    float total = 0.0;
    [[unroll]] for (uint i = 0u; i < 128u / SUBGROUP_SIZE; i++) {
        total += turbo_kv_sh_partial[i];
    }
    return total;
}

float turbo_kv_workgroup_max(float val) {
    float partial = subgroupMax(val);
    if (subgroupElect()) turbo_kv_sh_partial[gl_SubgroupID] = partial;
    barrier();
    float m = turbo_kv_sh_partial[0];
    [[unroll]] for (uint i = 1u; i < 128u / SUBGROUP_SIZE; i++) {
        m = max(m, turbo_kv_sh_partial[i]);
    }
    return m;
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
