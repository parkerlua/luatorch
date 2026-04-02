#include "../../tensor.h"
#include <cuda_runtime.h>
#include <stdio.h>

// forward declaration
extern "C" Tensor* tensor_new_cuda(int64_t* shape, int ndim, DType dtype);

#define BLOCK_SIZE 256

static int num_blocks(int64_t n) {
    return (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

#define CUDA_KERNEL_CHECK() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "luatorch cuda kernel error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        return NULL; \
    } \
} while(0)

// activation kernels

__global__ void relu_kernel(float* a, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] > 0.0f ? a[i] : 0.0f;
}

__global__ void sigmoid_kernel(float* a, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / (1.0f + expf(-a[i]));
}

__global__ void tanh_kernel(float* a, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanhf(a[i]);
}

__global__ void gelu_kernel(float* a, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

__global__ void silu_kernel(float* a, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        out[i] = x / (1.0f + expf(-x));
    }
}

__global__ void gt_scalar_kernel(float* a, float scalar, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] > scalar ? 1.0f : 0.0f;
}

// softmax kernel for 2D tensors
// each block handles one row
__global__ void softmax_kernel(float* a, float* out, int64_t rows, int64_t cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float* row_in  = a   + row * cols;
    float* row_out = out + row * cols;

    // find max for numerical stability
    float max_val = row_in[0];
    for (int64_t c = 1; c < cols; c++) {
        if (row_in[c] > max_val) max_val = row_in[c];
    }

    // exp and sum
    float sum = 0.0f;
    for (int64_t c = 0; c < cols; c++) {
        row_out[c] = expf(row_in[c] - max_val);
        sum += row_out[c];
    }

    // normalize
    for (int64_t c = 0; c < cols; c++) {
        row_out[c] /= sum;
    }
}

// cuda activation wrappers

extern "C" Tensor* tensor_relu_cuda(Tensor* a) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    relu_kernel<<<num_blocks(a->numel), BLOCK_SIZE>>>(a->cuda_data, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

extern "C" Tensor* tensor_sigmoid_cuda(Tensor* a) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    sigmoid_kernel<<<num_blocks(a->numel), BLOCK_SIZE>>>(a->cuda_data, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

extern "C" Tensor* tensor_tanh_cuda(Tensor* a) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    tanh_kernel<<<num_blocks(a->numel), BLOCK_SIZE>>>(a->cuda_data, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

extern "C" Tensor* tensor_gelu_cuda(Tensor* a) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    gelu_kernel<<<num_blocks(a->numel), BLOCK_SIZE>>>(a->cuda_data, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

extern "C" Tensor* tensor_silu_cuda(Tensor* a) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    silu_kernel<<<num_blocks(a->numel), BLOCK_SIZE>>>(a->cuda_data, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

extern "C" Tensor* tensor_gt_scalar_cuda(Tensor* a, float scalar) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    gt_scalar_kernel<<<num_blocks(a->numel), BLOCK_SIZE>>>(a->cuda_data, scalar, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

extern "C" Tensor* tensor_softmax_cuda(Tensor* a) {
    if (!a) return NULL;

    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->ndim == 1) {
        // treat as single row
        softmax_kernel<<<1, 1>>>(a->cuda_data, out->cuda_data, 1, a->numel);
    } else if (a->ndim == 2) {
        // one block per row
        softmax_kernel<<<a->shape[0], 1>>>(a->cuda_data, out->cuda_data, a->shape[0], a->shape[1]);
    } else {
        fprintf(stderr, "luatorch error: cuda softmax only supports 1D and 2D\n");
        return NULL;
    }

    CUDA_KERNEL_CHECK();
    return out;
}
