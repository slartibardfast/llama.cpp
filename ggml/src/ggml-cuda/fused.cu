#include "fused.cuh"
#include "ggml-fusion.h"

static __global__ void fused_silu_mul_kernel(const float * x, const float * y, float * dst, const int64_t k) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    const float xi = x[i];
    dst[i] = (xi / (1.0f + expf(-xi))) * y[i];
}

static __global__ void fused_sigmoid_mul_kernel(const float * x, const float * y, float * dst, const int64_t k) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = (1.0f / (1.0f + expf(-x[i]))) * y[i];
}

static __global__ void fused_gate_prep_kernel(const float * alpha, const float * dt_bias, const float * ssm_a,
                                              float * dst, const int64_t k, const int num_v_heads) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    const int h = (int)(i % num_v_heads);
    const float xv = alpha[i] + dt_bias[h];
    const float sp = (xv > 20.0f) ? xv : logf(1.0f + expf(xv));
    dst[i] = sp * ssm_a[h];
}

void ggml_cuda_op_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int32_t fusion_id = ggml_get_op_params_i32(dst, 0);

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t k = ggml_nelements(dst);
    const int64_t num_blocks = (k + CUDA_FUSED_BLOCK_SIZE - 1) / CUDA_FUSED_BLOCK_SIZE;
    cudaStream_t stream = ctx.stream();

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float *) dst->data;

    switch (fusion_id) {
        case GGML_FUSION_SILU_MUL:
            GGML_ASSERT(ggml_are_same_shape(src0, src1));
            fused_silu_mul_kernel<<<num_blocks, CUDA_FUSED_BLOCK_SIZE, 0, stream>>>(src0_d, src1_d, dst_d, k);
            break;
        case GGML_FUSION_SIGMOID_MUL:
            GGML_ASSERT(ggml_are_same_shape(src0, src1));
            fused_sigmoid_mul_kernel<<<num_blocks, CUDA_FUSED_BLOCK_SIZE, 0, stream>>>(src0_d, src1_d, dst_d, k);
            break;
        case GGML_FUSION_GATE_PREP: {
            const ggml_tensor * src2 = dst->src[2];
            GGML_ASSERT(src2 && src2->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(src2));
            const int num_v_heads = ggml_get_op_params_i32(dst, 1);
            GGML_ASSERT(num_v_heads > 0);
            const float * src2_d = (const float *) src2->data;
            fused_gate_prep_kernel<<<num_blocks, CUDA_FUSED_BLOCK_SIZE, 0, stream>>>(src0_d, src1_d, src2_d, dst_d, k, num_v_heads);
            break;
        }
        default:
            GGML_ABORT("ggml_cuda_op_fused: unsupported fusion_id %d", fusion_id);
    }
}
