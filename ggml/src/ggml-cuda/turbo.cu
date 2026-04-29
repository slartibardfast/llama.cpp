#include "turbo.cuh"
#include <mutex>

// One row × one column per CUDA block; 32 threads × 4 values per 128-element
// block. Forward RHT in-warp via __shfl_xor_sync stages 1..16 + cross-quartet
// butterfly on the 4 lane-local values. Codebook in a per-device buffer so
// imatrix-tuned centroids can replace the published Lloyd-Max defaults.

#define TKB_BLOCK 128
#define TKB_RSQRT128 0.08838834764831845f
#define TKB_SEED 0x12345678u

// Per-device codebook buffers, lazily populated. Index: bits-2 -> [0..3].
static float * g_turbo_cb[GGML_CUDA_MAX_DEVICES][4] = {};
static std::once_flag g_turbo_cb_init[GGML_CUDA_MAX_DEVICES];

static void turbo_codebook_init(int device) {
    std::call_once(g_turbo_cb_init[device], [device]() {
        ggml_cuda_set_device(device);
        const float * defaults[4] = {
            turbo_codebook_2bit, turbo_codebook_3bit,
            turbo_kv_4b_codebook, turbo_codebook_5bit  // 4b -> shared with kv_4b
        };
        for (int i = 0; i < 4; ++i) {
            const size_t n = (size_t)1 << (i + 2);
            CUDA_CHECK(cudaMalloc(&g_turbo_cb[device][i], n * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(g_turbo_cb[device][i], defaults[i],
                                   n * sizeof(float), cudaMemcpyHostToDevice));
        }
    });
}

void ggml_cuda_set_turbo_codebook(int device, int bits, const float * centroids) {
    GGML_ASSERT(bits >= 2 && bits <= 5);
    turbo_codebook_init(device);
    const size_t n = (size_t)1 << bits;
    ggml_cuda_set_device(device);
    CUDA_CHECK(cudaMemcpy(g_turbo_cb[device][bits - 2], centroids,
                           n * sizeof(float), cudaMemcpyHostToDevice));
}

__device__ __forceinline__ float turbo_sign(uint32_t idx) {
    uint32_t h = (TKB_SEED ^ idx) * 2654435761u;
    return (h & 1u) ? 1.0f : -1.0f;
}

// Read BITS-wide index at element position elem from a packed byte array.
template <int BITS>
__device__ __forceinline__ uint32_t turbo_unpack(const uint8_t * qs, uint32_t elem) {
    const uint32_t bit_off = elem * BITS;
    const uint32_t byte    = bit_off >> 3;
    const uint32_t shift   = bit_off & 7u;
    const uint32_t mask    = (1u << BITS) - 1u;
    uint32_t v = ((uint32_t)qs[byte]) >> shift;
    if (shift + BITS > 8u) v |= ((uint32_t)qs[byte + 1]) << (8u - shift);
    return v & mask;
}

// Wave32 forward RHT on 4 lane-local values (one CUDA warp = 32 lanes).
__device__ __forceinline__ void turbo_forward_rht_w32(
    float & v0, float & v1, float & v2, float & v3, uint32_t tid)
{
    v0 *= turbo_sign(tid);
    v1 *= turbo_sign(tid + 32u);
    v2 *= turbo_sign(tid + 64u);
    v3 *= turbo_sign(tid + 96u);
    // Strides 1, 2, 4, 8, 16 — within warp.
    #pragma unroll
    for (uint32_t s = 1u; s <= 16u; s <<= 1) {
        float o0 = __shfl_xor_sync(0xFFFFFFFFu, v0, s);
        float o1 = __shfl_xor_sync(0xFFFFFFFFu, v1, s);
        float o2 = __shfl_xor_sync(0xFFFFFFFFu, v2, s);
        float o3 = __shfl_xor_sync(0xFFFFFFFFu, v3, s);
        const bool hi = (tid & s) != 0u;
        v0 = hi ? (o0 - v0) : (v0 + o0);
        v1 = hi ? (o1 - v1) : (v1 + o1);
        v2 = hi ? (o2 - v2) : (v2 + o2);
        v3 = hi ? (o3 - v3) : (v3 + o3);
    }
    // Cross-quartet butterfly on the 4 local values.
    float t0, t1;
    t0 = v0; t1 = v1; v0 = t0 + t1; v1 = t0 - t1;
    t0 = v2; t1 = v3; v2 = t0 + t1; v3 = t0 - t1;
    t0 = v0; t1 = v2; v0 = t0 + t1; v2 = t0 - t1;
    t0 = v1; t1 = v3; v1 = t0 + t1; v3 = t0 - t1;
    v0 *= TKB_RSQRT128;
    v1 *= TKB_RSQRT128;
    v2 *= TKB_RSQRT128;
    v3 *= TKB_RSQRT128;
}

// fp16 → f32 (matches the host-side fp16 layout used by block_turbo_*b).
__device__ __forceinline__ float turbo_fp16_to_f32(uint16_t h) {
    return __half2float(*(const __half *)&h);
}

// One block = one row of dst. ne00 (= K) must be a multiple of TKB_BLOCK.
template <int BITS, int BLOCK_BYTES>
static __global__ void mul_mat_vec_turbo_kernel(
    const uint8_t * __restrict__ A,    // weights, packed
    const float   * __restrict__ B,    // f32 activation
    float         * __restrict__ D,    // f32 output
    const float   * __restrict__ CB,   // codebook
    const int64_t   ncols,             // K
    const int64_t   nrows,             // M
    const int64_t   stride_b,          // = ncols typically
    const int64_t   stride_d,          // = nrows typically (one column)
    const int64_t   batch_stride_a,    // bytes
    const int64_t   batch_stride_b,    // f32 elements
    const int64_t   batch_stride_d,    // f32 elements
    const int       broadcast2,
    const int       broadcast3,
    const int64_t   ne02,
    const int64_t   ne12)
{
    const uint32_t tid = threadIdx.x;
    const int64_t row  = blockIdx.x;
    const int64_t batch = blockIdx.y;
    if (row >= nrows) return;

    int64_t batch_a = 0;
    if (batch != 0) {
        const int64_t i13 = batch / ne12;
        const int64_t i12 = batch % ne12;
        const int64_t i03 = i13 / broadcast3;
        const int64_t i02 = i12 / broadcast2;
        batch_a = i03 * ne02 + i02;
    }

    const int64_t bytes_per_row = (ncols / TKB_BLOCK) * BLOCK_BYTES;
    const uint8_t * A_row  = A + batch_a * bytes_per_row * nrows + row * bytes_per_row;
    const float   * B_base = B + batch * batch_stride_b;

    const int64_t n_blocks = ncols / TKB_BLOCK;
    float row_acc = 0.0f;

    for (int64_t blk = 0; blk < n_blocks; ++blk) {
        const uint8_t * blk_ptr = A_row + blk * BLOCK_BYTES;
        const uint16_t norm_bits    = (uint16_t)blk_ptr[0] | ((uint16_t)blk_ptr[1] << 8);
        const uint16_t inv_std_bits = (uint16_t)blk_ptr[2] | ((uint16_t)blk_ptr[3] << 8);
        const float norm    = turbo_fp16_to_f32(norm_bits);
        const float inv_std = turbo_fp16_to_f32(inv_std_bits);
        const float rcp_inv_std = 1.0f / fmaxf(inv_std, 1e-10f);

        const uint8_t * qs = blk_ptr + 4;
        const float * b_blk = B_base + (size_t)blk * TKB_BLOCK;

        // Load 4 activation values (lane-local layout matching wave32 RHT).
        float a0 = b_blk[tid];
        float a1 = b_blk[tid + 32u];
        float a2 = b_blk[tid + 64u];
        float a3 = b_blk[tid + 96u];
        turbo_forward_rht_w32(a0, a1, a2, a3, tid);

        const float w0 = CB[turbo_unpack<BITS>(qs, tid)]        * rcp_inv_std;
        const float w1 = CB[turbo_unpack<BITS>(qs, tid + 32u)]  * rcp_inv_std;
        const float w2 = CB[turbo_unpack<BITS>(qs, tid + 64u)]  * rcp_inv_std;
        const float w3 = CB[turbo_unpack<BITS>(qs, tid + 96u)]  * rcp_inv_std;

        float local_dot = w0 * a0 + w1 * a1 + w2 * a2 + w3 * a3;
        // Warp reduce.
        #pragma unroll
        for (uint32_t s = 16u; s > 0u; s >>= 1) {
            local_dot += __shfl_xor_sync(0xFFFFFFFFu, local_dot, s);
        }
        row_acc += local_dot * norm;
    }

    if (tid == 0u) {
        D[batch * batch_stride_d + row] = row_acc;
    }
}

bool ggml_cuda_can_mul_mat_turbo(const ggml_tensor * src0) {
    switch (src0->type) {
        case GGML_TYPE_TURBO_2B:
        case GGML_TYPE_TURBO_3B:
        case GGML_TYPE_TURBO_4B:
        case GGML_TYPE_TURBO_5B:
            return true;
        default:
            return false;
    }
}

void ggml_cuda_mul_mat_turbo(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst)
{
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] % TKB_BLOCK == 0);
    GGML_ASSERT(src1->ne[1] == 1); // mul_mat_vec only

    turbo_codebook_init(ctx.device);

    int bits = 0;
    int block_bytes = 0;
    int cb_idx = 0;
    switch (src0->type) {
        case GGML_TYPE_TURBO_2B: bits = 2; block_bytes = 4 + (TKB_BLOCK * 2 / 8); cb_idx = 0; break;
        case GGML_TYPE_TURBO_3B: bits = 3; block_bytes = 4 + (TKB_BLOCK * 3 / 8); cb_idx = 1; break;
        case GGML_TYPE_TURBO_4B: bits = 4; block_bytes = 4 + (TKB_BLOCK * 4 / 8); cb_idx = 2; break;
        case GGML_TYPE_TURBO_5B: bits = 5; block_bytes = 4 + (TKB_BLOCK * 5 / 8); cb_idx = 3; break;
        default: GGML_ABORT("ggml_cuda_mul_mat_turbo: unsupported type");
    }

    const float * cb = g_turbo_cb[ctx.device][cb_idx];

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = src0->ne[1];
    const int64_t ne02  = src0->ne[2];
    const int64_t ne03  = src0->ne[3];
    const int64_t ne12  = src1->ne[2];
    const int64_t ne13  = src1->ne[3];
    const int     broadcast2 = (int)(ne12 / ne02);
    const int     broadcast3 = (int)(ne13 / ne03);

    const int64_t batches = ne12 * ne13;

    const dim3 grid((unsigned)nrows, (unsigned)batches, 1);
    const dim3 block(32, 1, 1);
    cudaStream_t stream = ctx.stream();

    const uint8_t * A = (const uint8_t *)src0->data;
    const float   * B = (const float   *)src1->data;
    float         * D = (float         *)dst->data;

    const int64_t batch_stride_a = (ncols / TKB_BLOCK) * block_bytes * nrows;
    const int64_t batch_stride_b = src1->nb[2] / sizeof(float);
    const int64_t batch_stride_d =  dst->nb[2] / sizeof(float);
    const int64_t stride_b       = src1->nb[1] / sizeof(float);
    const int64_t stride_d       =  dst->nb[1] / sizeof(float);

    switch (bits) {
        case 2: mul_mat_vec_turbo_kernel<2, 4 + TKB_BLOCK*2/8><<<grid, block, 0, stream>>>(
                    A, B, D, cb, ncols, nrows, stride_b, stride_d,
                    batch_stride_a, batch_stride_b, batch_stride_d,
                    broadcast2, broadcast3, ne02, ne12); break;
        case 3: mul_mat_vec_turbo_kernel<3, 4 + TKB_BLOCK*3/8><<<grid, block, 0, stream>>>(
                    A, B, D, cb, ncols, nrows, stride_b, stride_d,
                    batch_stride_a, batch_stride_b, batch_stride_d,
                    broadcast2, broadcast3, ne02, ne12); break;
        case 4: mul_mat_vec_turbo_kernel<4, 4 + TKB_BLOCK*4/8><<<grid, block, 0, stream>>>(
                    A, B, D, cb, ncols, nrows, stride_b, stride_d,
                    batch_stride_a, batch_stride_b, batch_stride_d,
                    broadcast2, broadcast3, ne02, ne12); break;
        case 5: mul_mat_vec_turbo_kernel<5, 4 + TKB_BLOCK*5/8><<<grid, block, 0, stream>>>(
                    A, B, D, cb, ncols, nrows, stride_b, stride_d,
                    batch_stride_a, batch_stride_b, batch_stride_d,
                    broadcast2, broadcast3, ne02, ne12); break;
    }
}

// =============================================================================
// dequant_turbo_kernel<BITS>: TURBO_*B (packed) → fp16 (contiguous, row-major).
// =============================================================================
// One CUDA block per 128-element source block; 32 threads (one warp); 4
// lane-local values per block. Inverse RHT (forward butterfly + sign-flip
// AFTER butterfly), scale by norm * 1/sqrt(128), write fp16. Used as the
// dequant prelude to cuBLAS HGEMM via ggml_get_to_fp16_cuda for n>=2 mat-mul.
// Source data is contiguous packed blocks; output is contiguous fp16.

template <int BITS, int BLOCK_BYTES>
static __global__ void dequant_turbo_kernel(
    const uint8_t * __restrict__ A,    // packed weights, contiguous
    half          * __restrict__ Y,    // fp16 destination, contiguous
    const float   * __restrict__ CB,   // codebook
    const int64_t   n_blocks)
{
    const uint32_t tid = threadIdx.x;
    const int64_t blk  = (int64_t)blockIdx.x + (int64_t)blockIdx.y * gridDim.x;
    if (blk >= n_blocks) return;

    const uint8_t * blk_ptr = A + blk * BLOCK_BYTES;
    const uint16_t norm_bits    = (uint16_t)blk_ptr[0] | ((uint16_t)blk_ptr[1] << 8);
    const uint16_t inv_std_bits = (uint16_t)blk_ptr[2] | ((uint16_t)blk_ptr[3] << 8);
    const float norm    = turbo_fp16_to_f32(norm_bits);
    const float inv_std = turbo_fp16_to_f32(inv_std_bits);
    const float rcp_inv_std = 1.0f / fmaxf(inv_std, 1e-10f);

    const uint8_t * qs = blk_ptr + 4;

    // Codebook lookup → 4 lane-local values, scaled by 1/inv_std.
    float v0 = CB[turbo_unpack<BITS>(qs, tid)]        * rcp_inv_std;
    float v1 = CB[turbo_unpack<BITS>(qs, tid + 32u)]  * rcp_inv_std;
    float v2 = CB[turbo_unpack<BITS>(qs, tid + 64u)]  * rcp_inv_std;
    float v3 = CB[turbo_unpack<BITS>(qs, tid + 96u)]  * rcp_inv_std;

    // Inverse RHT: butterfly first (5 in-warp stages + cross-quartet),
    // then scale by norm/sqrt(128), then sign-flip.
    #pragma unroll
    for (uint32_t s = 1u; s <= 16u; s <<= 1) {
        float o0 = __shfl_xor_sync(0xFFFFFFFFu, v0, s);
        float o1 = __shfl_xor_sync(0xFFFFFFFFu, v1, s);
        float o2 = __shfl_xor_sync(0xFFFFFFFFu, v2, s);
        float o3 = __shfl_xor_sync(0xFFFFFFFFu, v3, s);
        const bool hi = (tid & s) != 0u;
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

    const float scale = norm * TKB_RSQRT128;
    v0 *= scale * turbo_sign(tid);
    v1 *= scale * turbo_sign(tid + 32u);
    v2 *= scale * turbo_sign(tid + 64u);
    v3 *= scale * turbo_sign(tid + 96u);

    half * y_blk = Y + blk * TKB_BLOCK;
    y_blk[tid]        = __float2half(v0);
    y_blk[tid + 32u]  = __float2half(v1);
    y_blk[tid + 64u]  = __float2half(v2);
    y_blk[tid + 96u]  = __float2half(v3);
}

template <int BITS, int BLOCK_BYTES>
static void dequant_row_turbo_cuda_impl(
    const void * vx, half * y, int64_t k, cudaStream_t stream)
{
    GGML_ASSERT(k % TKB_BLOCK == 0);
    const int device = ggml_cuda_get_device();
    turbo_codebook_init(device);
    const float * cb = g_turbo_cb[device][BITS - 2];
    const int64_t n_blocks = k / TKB_BLOCK;
    // Use 2D grid to dodge the 65535 limit on grid.x for very large k.
    const unsigned gx = (unsigned)std::min<int64_t>(n_blocks, 65535);
    const unsigned gy = (unsigned)((n_blocks + gx - 1) / gx);
    dequant_turbo_kernel<BITS, BLOCK_BYTES><<<dim3(gx, gy, 1), dim3(32, 1, 1), 0, stream>>>(
        (const uint8_t *)vx, y, cb, n_blocks);
}

void dequant_row_turbo_2b_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequant_row_turbo_cuda_impl<2, 4 + TKB_BLOCK*2/8>(vx, y, k, stream);
}
void dequant_row_turbo_3b_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequant_row_turbo_cuda_impl<3, 4 + TKB_BLOCK*3/8>(vx, y, k, stream);
}
void dequant_row_turbo_4b_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequant_row_turbo_cuda_impl<4, 4 + TKB_BLOCK*4/8>(vx, y, k, stream);
}
void dequant_row_turbo_5b_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequant_row_turbo_cuda_impl<5, 4 + TKB_BLOCK*5/8>(vx, y, k, stream);
}

bool ggml_cuda_should_dequant_turbo(int cc, int64_t n) {
    if (n < 2) return false;
    // sm_80+ (Ampere/Ada/Hopper/Blackwell) and AMD CDNA: tensor cores cheap,
    // dequant+HGEMM beats fused at any n>=2.
    if (cc >= GGML_CUDA_CC_AMPERE) return true;
#ifdef GGML_USE_HIP
    // CDNA gfx908+ has MFMA via hipBLAS; keep n>=2 threshold.
    if (cc >= GGML_CUDA_CC_CDNA) return true;
    // RDNA3+ via WMMA: shift threshold to n>=4.
    if (cc >= GGML_CUDA_CC_RDNA3) return n >= 4;
    // Older AMD: fall back to SGEMM at n>=8.
    return n >= 8;
#else
    // Volta sm_70 / Turing sm_75: HMMA without cp.async; n>=4 threshold.
    return n >= 4;
#endif
}
