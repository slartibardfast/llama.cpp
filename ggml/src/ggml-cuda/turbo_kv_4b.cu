#include "turbo_kv_4b.cuh"

// TURBO_KV_4B CUDA kernels. Mirrors the Vulkan port at
// ggml/src/ggml-vulkan/vulkan-shaders/{turbo_kv_4b_rht.glsl, ...}.
// One CUDA block per 128-element TURBO_KV_4B block, 128 threads, one element
// per lane. The 7-stage Walsh–Hadamard butterfly splits as:
//   stages 1..16: __shfl_xor_sync within a warp (warpSize=32, no barrier).
//   stages 32, 64: shared memory + __syncthreads.

#define TKV_BLOCK 128
#define TKV_RSQRT128 0.08838834764f
#define TKV_CENT_MAX 2.7326f
#define TKV_SEED 0x12345678u

__device__ __forceinline__ float tkv_sign(uint32_t idx) {
    uint32_t h = (TKV_SEED ^ idx) * 2654435761u;
    return (h & 1u) ? 1.0f : -1.0f;
}

__device__ __constant__ float tkv_codebook[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
};

__device__ __forceinline__ uint32_t tkv_nearest(float x) {
    uint32_t best = 0u;
    float best_d = fabsf(x - tkv_codebook[0]);
    #pragma unroll
    for (uint32_t c = 1u; c < 16u; ++c) {
        float d = fabsf(x - tkv_codebook[c]);
        if (d < best_d) { best = c; best_d = d; }
    }
    return best;
}

__device__ __forceinline__ float tkv_unpack(uint32_t byte_val, uint32_t elem, float rcp_inv_std) {
    uint32_t idx = (elem & 1u) == 0u ? (byte_val & 0xFu) : (byte_val >> 4);
    return tkv_codebook[idx] * rcp_inv_std;
}

// In-place 7-stage FWHT on a 128-element block. lds must be 128 floats shared.
__device__ __forceinline__ void tkv_fwht(float & val, uint32_t tid, float * lds) {
    // Stages 1, 2, 4, 8, 16 — within warp.
    #pragma unroll
    for (uint32_t s = 1u; s < 32u; s <<= 1) {
        float p = __shfl_xor_sync(0xFFFFFFFFu, val, s);
        val = ((tid & s) == 0u) ? (val + p) : (p - val);
    }
    // Stages 32, 64 — across warps via shared memory.
    #pragma unroll
    for (uint32_t s = 32u; s < 128u; s <<= 1) {
        __syncthreads();
        lds[tid] = val;
        __syncthreads();
        float p = lds[tid ^ s];
        val = ((tid & s) == 0u) ? (val + p) : (p - val);
    }
}

// Workgroup sum/max across all 128 lanes. lds must be 128 floats shared
// (we reuse the butterfly buffer; caller ensures no pending data).
__device__ __forceinline__ float tkv_block_sum(float val, float * lds, uint32_t tid) {
    // Warp reduce.
    #pragma unroll
    for (uint32_t s = 16u; s > 0u; s >>= 1) val += __shfl_xor_sync(0xFFFFFFFFu, val, s);
    __syncthreads();
    if ((tid & 31u) == 0u) lds[tid >> 5] = val;
    __syncthreads();
    float total = lds[0] + lds[1] + lds[2] + lds[3];
    return total;
}

__device__ __forceinline__ float tkv_block_max(float val, float * lds, uint32_t tid) {
    #pragma unroll
    for (uint32_t s = 16u; s > 0u; s >>= 1) val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFu, val, s));
    __syncthreads();
    if ((tid & 31u) == 0u) lds[tid >> 5] = val;
    __syncthreads();
    float m = fmaxf(fmaxf(lds[0], lds[1]), fmaxf(lds[2], lds[3]));
    return m;
}

// =============================================================================
// CPY F32 -> TURBO_KV_4B
// =============================================================================
// Generic strided-source kernel. block_id maps to a flat 128-element block of
// the *source* tensor; we decompose into (i00, i01, i02, i03) using src ne/nb
// and gather 128 contiguous-in-i00 elements via stride nb00.

static __global__ void cpy_f32_turbo_kv_4b_kernel(
    const char * __restrict__ cx, char * __restrict__ cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13)
{
    const uint32_t tid = threadIdx.x;
    const int64_t block_id = (int64_t)blockIdx.x;
    const int64_t flat = block_id * TKV_BLOCK;
    if (flat >= ne) return;

    // Source index decomposition.
    const int64_t i03 = flat / (ne00 * ne01 * ne02);
    int64_t rem = flat - i03 * ne00 * ne01 * ne02;
    const int64_t i02 = rem / (ne00 * ne01); rem -= i02 * ne00 * ne01;
    const int64_t i01 = rem / ne00;
    const int64_t i00 = rem - i01 * ne00;

    const char * x_base = cx + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
    float val = *(const float *)(x_base + tid * nb00);

    __shared__ float lds[TKV_BLOCK];

    // L2 normalize.
    float sum_sq = tkv_block_sum(val * val, lds, tid);
    float norm = sqrtf(sum_sq);
    val *= 1.0f / fmaxf(norm, 1e-10f);

    // Sign-flip then butterfly then 1/sqrt(N).
    val *= tkv_sign(tid);
    tkv_fwht(val, tid, lds);
    val *= TKV_RSQRT128;

    // Per-block inv_std.
    float max_abs = tkv_block_max(fabsf(val), lds, tid);
    float inv_std = TKV_CENT_MAX / fmaxf(max_abs, 1e-10f);

    uint32_t idx = tkv_nearest(val * inv_std);

    // Destination block. dst is a flat array of block_turbo_kv_4b; nb10 is the
    // block stride (= sizeof(block_turbo_kv_4b)) when row-contiguous.
    const int64_t i13 = flat / (ne10 * ne11 * ne12);
    rem = flat - i13 * ne10 * ne11 * ne12;
    const int64_t i12 = rem / (ne10 * ne11); rem -= i12 * ne10 * ne11;
    const int64_t i11 = rem / ne10;
    const int64_t i10 = rem - i11 * ne10;
    const int64_t dst_off = (i10 / TKV_BLOCK) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;
    block_turbo_kv_4b * blk = (block_turbo_kv_4b *)(cdst + dst_off);

    // Pack pairs of 4-bit indices.
    uint32_t partner = __shfl_xor_sync(0xFFFFFFFFu, idx, 1);
    if ((tid & 1u) == 0u) {
        blk->mse_indices[tid >> 1] = (uint8_t)(idx | (partner << 4));
    }
    if (tid == 0u) {
        blk->norm = norm;
        blk->inv_std = inv_std;
    }
}

void ggml_cuda_cpy_f32_turbo_kv_4b(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream)
{
    GGML_ASSERT(ne % TKV_BLOCK == 0);
    const int64_t num_blocks = ne / TKV_BLOCK;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_turbo_kv_4b_kernel<<<(unsigned)num_blocks, TKV_BLOCK, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

// =============================================================================
// CPY TURBO_KV_4B -> F32
// =============================================================================
static __global__ void cpy_turbo_kv_4b_f32_kernel(
    const char * __restrict__ cx, char * __restrict__ cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13)
{
    const uint32_t tid = threadIdx.x;
    const int64_t block_id = (int64_t)blockIdx.x;
    const int64_t flat = block_id * TKV_BLOCK;
    if (flat >= ne) return;

    // Source: quantized block.
    const int64_t i03 = flat / (ne00 * ne01 * ne02);
    int64_t rem = flat - i03 * ne00 * ne01 * ne02;
    const int64_t i02 = rem / (ne00 * ne01); rem -= i02 * ne00 * ne01;
    const int64_t i01 = rem / ne00;
    const int64_t i00 = rem - i01 * ne00;
    const int64_t src_off = (i00 / TKV_BLOCK) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
    const block_turbo_kv_4b * blk = (const block_turbo_kv_4b *)(cx + src_off);

    const float norm = blk->norm;
    const float inv_std = blk->inv_std;
    const float rcp_inv_std = 1.0f / fmaxf(inv_std, 1e-10f);
    float val = tkv_unpack((uint32_t)blk->mse_indices[tid >> 1], tid, rcp_inv_std);

    __shared__ float lds[TKV_BLOCK];

    // Inverse RHT: butterfly → normalize → sign.
    tkv_fwht(val, tid, lds);
    val *= norm * TKV_RSQRT128 * tkv_sign(tid);

    // Destination: contiguous f32.
    const int64_t i13 = flat / (ne10 * ne11 * ne12);
    rem = flat - i13 * ne10 * ne11 * ne12;
    const int64_t i12 = rem / (ne10 * ne11); rem -= i12 * ne10 * ne11;
    const int64_t i11 = rem / ne10;
    const int64_t i10 = rem - i11 * ne10;
    const int64_t dst_off = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;
    *(float *)(cdst + dst_off + tid * nb10) = val;
}

void ggml_cuda_cpy_turbo_kv_4b_f32(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream)
{
    GGML_ASSERT(ne % TKV_BLOCK == 0);
    const int64_t num_blocks = ne / TKV_BLOCK;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_turbo_kv_4b_f32_kernel<<<(unsigned)num_blocks, TKV_BLOCK, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

// =============================================================================
// SET_ROWS F32 -> TURBO_KV_4B
// =============================================================================
// dst tensor: TURBO_KV_4B blocks. src0 (vx) is f32 source. src1 (vidx) is the
// row-index tensor (i32 or i64). One CUDA block per 128-element source block.

template <typename idx_t>
static __global__ void set_rows_turbo_kv_4b_kernel(
    const char * __restrict__ vx,    // src0: f32 source
    const char * __restrict__ vidx,  // src1: row index
    char * __restrict__ vdst,        // dst: turbo_kv_4b
    const int64_t ne,                // total f32 elements in src0
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12,
    const int64_t nb20, const int64_t nb21, const int64_t nb22, const int64_t nb23)
{
    const uint32_t tid = threadIdx.x;
    const int64_t wg = (int64_t)blockIdx.x;
    const int64_t flat = wg * TKV_BLOCK;
    if (flat >= ne) return;

    // Decompose flat index into 4D source coords (units = elements of src0).
    const int64_t i03 = flat / (ne00 * ne01 * ne02);
    int64_t rem = flat - i03 * ne00 * ne01 * ne02;
    const int64_t i02 = rem / (ne00 * ne01); rem -= i02 * ne00 * ne01;
    const int64_t i01 = rem / ne00;
    const int64_t i00 = rem - i01 * ne00;

    // Index tensor lookup: dst row = vidx[(i02 % ne11), i01]. ne12 = ne12 of
    // the dst block-row count; broadcast as in vk set_rows.
    const int64_t i10 = i01;
    const int64_t i11 = i02 % ne11;
    const int64_t i12 = i03 % ne12;
    const int64_t b_off = i12 * nb12 + i11 * nb11 + i10 * nb10;
    const int64_t dst_row = (int64_t) *(const idx_t *)(vidx + b_off);

    // Source f32 base.
    const char * x_base = vx + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
    float val = *(const float *)(x_base + tid * nb00);

    __shared__ float lds[TKV_BLOCK];

    float sum_sq = tkv_block_sum(val * val, lds, tid);
    float norm = sqrtf(sum_sq);
    val *= 1.0f / fmaxf(norm, 1e-10f);

    val *= tkv_sign(tid);
    tkv_fwht(val, tid, lds);
    val *= TKV_RSQRT128;

    float max_abs = tkv_block_max(fabsf(val), lds, tid);
    float inv_std = TKV_CENT_MAX / fmaxf(max_abs, 1e-10f);

    uint32_t idx = tkv_nearest(val * inv_std);

    const int64_t dst_off = i03 * nb23 + i02 * nb22 + dst_row * nb21
                          + (i00 / TKV_BLOCK) * nb20;
    block_turbo_kv_4b * blk = (block_turbo_kv_4b *)(vdst + dst_off);

    uint32_t partner = __shfl_xor_sync(0xFFFFFFFFu, idx, 1);
    if ((tid & 1u) == 0u) {
        blk->mse_indices[tid >> 1] = (uint8_t)(idx | (partner << 4));
    }
    if (tid == 0u) {
        blk->norm = norm;
        blk->inv_std = inv_std;
    }
}

void ggml_cuda_op_set_rows_turbo_kv_4b(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_TURBO_KV_4B);
    GGML_ASSERT(src0->ne[0] % TKV_BLOCK == 0);

    const int64_t ne = ggml_nelements(src0);
    const int64_t num_blocks = ne / TKV_BLOCK;

    cudaStream_t stream = ctx.stream();
    if (src1->type == GGML_TYPE_I64) {
        set_rows_turbo_kv_4b_kernel<int64_t><<<(unsigned)num_blocks, TKV_BLOCK, 0, stream>>>(
            (const char *)src0->data, (const char *)src1->data, (char *)dst->data,
            ne,
            src0->ne[0], src0->ne[1], src0->ne[2],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0], src1->ne[1], src1->ne[2],
            src1->nb[0], src1->nb[1], src1->nb[2],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    } else {
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        set_rows_turbo_kv_4b_kernel<int32_t><<<(unsigned)num_blocks, TKV_BLOCK, 0, stream>>>(
            (const char *)src0->data, (const char *)src1->data, (char *)dst->data,
            ne,
            src0->ne[0], src0->ne[1], src0->ne[2],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0], src1->ne[1], src1->ne[2],
            src1->nb[0], src1->nb[1], src1->nb[2],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    }
}
