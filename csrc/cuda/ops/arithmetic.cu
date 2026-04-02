#include "../../tensor.h"
#include <cuda_runtime.h>
#include <stdio.h>

// forward declaration
extern "C" Tensor* tensor_new_cuda(int64_t* shape, int ndim, DType dtype);

// standard block size for elementwise ops
// 256 threads per block is a good default for the 4090
#define BLOCK_SIZE 256

// helper to get number of blocks needed
static int num_blocks(int64_t n) {
    return (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

// check cuda errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "luatorch cuda error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        return NULL; \
    } \
} while(0)

#define CUDA_CHECK_VOID(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "luatorch cuda error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

#define CUDA_KERNEL_CHECK() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "luatorch cuda kernel error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        return NULL; \
    } \
} while(0)

#define CUDA_KERNEL_CHECK_VOID() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "luatorch cuda kernel error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

// elementwise binary kernels
// one thread per element, simplest possible approach

__global__ void add_kernel(float* a, float* b, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void sub_kernel(float* a, float* b, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

__global__ void mul_kernel(float* a, float* b, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void div_kernel(float* a, float* b, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / b[i];
}

// scalar kernels
__global__ void add_scalar_kernel(float* a, float scalar, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + scalar;
}

__global__ void mul_scalar_kernel(float* a, float scalar, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * scalar;
}

__global__ void div_scalar_kernel(float* a, float scalar, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / scalar;
}

__global__ void pow_scalar_kernel(float* a, float exp, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = powf(a[i], exp);
}

// unary kernels
__global__ void neg_kernel(float* a, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -a[i];
}

__global__ void abs_kernel(float* a, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fabsf(a[i]);
}

__global__ void sqrt_kernel(float* a, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sqrtf(a[i]);
}

__global__ void log_kernel(float* a, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = logf(a[i]);
}

__global__ void exp_kernel(float* a, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(a[i]);
}

// inplace kernels
__global__ void add_inplace_kernel(float* a, float* b, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void sub_inplace_kernel(float* a, float* b, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] -= b[i];
}

__global__ void mul_scalar_inplace_kernel(float* a, float scalar, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= scalar;
}

__global__ void add_scalar_inplace_kernel(float* a, float scalar, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += scalar;
}

// parallel reduction kernel for sum
// uses shared memory, each block reduces its chunk
// then we sum block results on cpu (fast enough for the final step)
__global__ void sum_kernel(float* data, float* block_sums, int64_t n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // load data into shared memory
    sdata[tid] = (i < n) ? data[i] : 0.0f;
    __syncthreads();

    // reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write block result
    if (tid == 0) block_sums[blockIdx.x] = sdata[0];
}

// parallel reduction for max
__global__ void max_kernel(float* data, float* block_maxs, int64_t n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? data[i] : -INFINITY;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) block_maxs[blockIdx.x] = sdata[0];
}

// parallel reduction for min
__global__ void min_kernel(float* data, float* block_mins, int64_t n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? data[i] : INFINITY;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) block_mins[blockIdx.x] = sdata[0];
}

// cuda elementwise binary ops

extern "C" Tensor* tensor_add_cuda(Tensor* a, Tensor* b) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    int blocks = num_blocks(a->numel);
    add_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, b->cuda_data, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

extern "C" Tensor* tensor_sub_cuda(Tensor* a, Tensor* b) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    int blocks = num_blocks(a->numel);
    sub_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, b->cuda_data, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

extern "C" Tensor* tensor_mul_cuda(Tensor* a, Tensor* b) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    int blocks = num_blocks(a->numel);
    mul_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, b->cuda_data, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

extern "C" Tensor* tensor_div_cuda(Tensor* a, Tensor* b) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    int blocks = num_blocks(a->numel);
    div_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, b->cuda_data, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

// cuda scalar ops

extern "C" Tensor* tensor_add_scalar_cuda(Tensor* a, float scalar) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    int blocks = num_blocks(a->numel);
    add_scalar_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, scalar, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

extern "C" Tensor* tensor_mul_scalar_cuda(Tensor* a, float scalar) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    int blocks = num_blocks(a->numel);
    mul_scalar_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, scalar, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

extern "C" Tensor* tensor_div_scalar_cuda(Tensor* a, float scalar) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    int blocks = num_blocks(a->numel);
    div_scalar_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, scalar, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

extern "C" Tensor* tensor_pow_scalar_cuda(Tensor* a, float exp) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    int blocks = num_blocks(a->numel);
    pow_scalar_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, exp, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

// cuda unary ops

extern "C" Tensor* tensor_neg_cuda(Tensor* a) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    int blocks = num_blocks(a->numel);
    neg_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

extern "C" Tensor* tensor_abs_cuda(Tensor* a) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    int blocks = num_blocks(a->numel);
    abs_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

extern "C" Tensor* tensor_sqrt_cuda(Tensor* a) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    int blocks = num_blocks(a->numel);
    sqrt_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

extern "C" Tensor* tensor_log_cuda(Tensor* a) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    int blocks = num_blocks(a->numel);
    log_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

extern "C" Tensor* tensor_exp_cuda(Tensor* a) {
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    int blocks = num_blocks(a->numel);
    exp_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, out->cuda_data, a->numel);
    CUDA_KERNEL_CHECK();
    return out;
}

// cuda reduction ops
// two pass: gpu reduces to block sums, cpu finishes the final reduction

// fix: added error checks on cudaMalloc and malloc in all reduction functions
// old code had no checks, leaked memory and crashed on allocation failure

extern "C" float tensor_sum_cuda(Tensor* a) {
    if (!a || a->numel == 0) return 0.0f;
    int blocks = num_blocks(a->numel);
    size_t smem = BLOCK_SIZE * sizeof(float);

    float* d_block_sums;
    if (cudaMalloc((void**)&d_block_sums, blocks * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "luatorch cuda error: reduction alloc failed\n");
        return 0.0f;
    }

    sum_kernel<<<blocks, BLOCK_SIZE, smem>>>(a->cuda_data, d_block_sums, a->numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "luatorch cuda kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_block_sums);
        return 0.0f;
    }

    float* h_block_sums = (float*)malloc(blocks * sizeof(float));
    if (!h_block_sums) {
        cudaFree(d_block_sums);
        return 0.0f;
    }
    cudaMemcpy(h_block_sums, d_block_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < blocks; i++) total += h_block_sums[i];

    free(h_block_sums);
    cudaFree(d_block_sums);
    return total;
}

extern "C" float tensor_mean_cuda(Tensor* a) {
    if (!a || a->numel == 0) return 0.0f;
    return tensor_sum_cuda(a) / (float)a->numel;
}

extern "C" float tensor_max_cuda(Tensor* a) {
    if (!a || a->numel == 0) return 0.0f;
    int blocks = num_blocks(a->numel);
    size_t smem = BLOCK_SIZE * sizeof(float);

    float* d_block_maxs;
    if (cudaMalloc((void**)&d_block_maxs, blocks * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "luatorch cuda error: reduction alloc failed\n");
        return 0.0f;
    }

    max_kernel<<<blocks, BLOCK_SIZE, smem>>>(a->cuda_data, d_block_maxs, a->numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "luatorch cuda kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_block_maxs);
        return 0.0f;
    }

    float* h_block_maxs = (float*)malloc(blocks * sizeof(float));
    if (!h_block_maxs) {
        cudaFree(d_block_maxs);
        return 0.0f;
    }
    cudaMemcpy(h_block_maxs, d_block_maxs, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float result = h_block_maxs[0];
    for (int i = 1; i < blocks; i++) {
        if (h_block_maxs[i] > result) result = h_block_maxs[i];
    }

    free(h_block_maxs);
    cudaFree(d_block_maxs);
    return result;
}

extern "C" float tensor_min_cuda(Tensor* a) {
    if (!a || a->numel == 0) return 0.0f;
    int blocks = num_blocks(a->numel);
    size_t smem = BLOCK_SIZE * sizeof(float);

    float* d_block_mins;
    if (cudaMalloc((void**)&d_block_mins, blocks * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "luatorch cuda error: reduction alloc failed\n");
        return 0.0f;
    }

    min_kernel<<<blocks, BLOCK_SIZE, smem>>>(a->cuda_data, d_block_mins, a->numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "luatorch cuda kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_block_mins);
        return 0.0f;
    }

    float* h_block_mins = (float*)malloc(blocks * sizeof(float));
    if (!h_block_mins) {
        cudaFree(d_block_mins);
        return 0.0f;
    }
    cudaMemcpy(h_block_mins, d_block_mins, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float result = h_block_mins[0];
    for (int i = 1; i < blocks; i++) {
        if (h_block_mins[i] < result) result = h_block_mins[i];
    }

    free(h_block_mins);
    cudaFree(d_block_mins);
    return result;
}

// cuda inplace ops

extern "C" void tensor_add_cuda_(Tensor* a, Tensor* b) {
    int blocks = num_blocks(a->numel);
    add_inplace_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, b->cuda_data, a->numel);
    CUDA_KERNEL_CHECK_VOID();
}

extern "C" void tensor_sub_cuda_(Tensor* a, Tensor* b) {
    int blocks = num_blocks(a->numel);
    sub_inplace_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, b->cuda_data, a->numel);
    CUDA_KERNEL_CHECK_VOID();
}

extern "C" void tensor_mul_scalar_cuda_(Tensor* a, float scalar) {
    int blocks = num_blocks(a->numel);
    mul_scalar_inplace_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, scalar, a->numel);
    CUDA_KERNEL_CHECK_VOID();
}

extern "C" void tensor_add_scalar_cuda_(Tensor* a, float scalar) {
    int blocks = num_blocks(a->numel);
    add_scalar_inplace_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, scalar, a->numel);
    CUDA_KERNEL_CHECK_VOID();
}

// broadcast add: a is [rows, cols], b is [cols]
// adds b to every row of a
__global__ void add_broadcast_kernel(float* a, float* b, float* out,
                                      int64_t rows, int64_t cols) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int64_t col = idx % cols;
    out[idx] = a[idx] + b[col];
}

extern "C" Tensor* tensor_add_broadcast_cuda(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    int64_t rows = a->shape[0];
    int64_t cols = a->shape[1];
    int blocks = num_blocks(rows * cols);

    add_broadcast_kernel<<<blocks, BLOCK_SIZE>>>(
        a->cuda_data, b->cuda_data, out->cuda_data, rows, cols);
    CUDA_KERNEL_CHECK();
    return out;
}

// perf fix: parallelize broadcast backward with shared memory atomics
// old code had one thread per column serially looping over all rows
// new code: each thread handles one element, uses atomicAdd to accumulate per-column
__global__ void add_broadcast_backward_kernel(float* grad, float* out,
                                                int64_t rows, int64_t cols) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int64_t col = idx % cols;
    atomicAdd(&out[col], grad[idx]);
}

extern "C" Tensor* tensor_add_broadcast_backward_cuda(Tensor* grad) {
    if (!grad) return NULL;
    int64_t rows = grad->shape[0];
    int64_t cols = grad->shape[1];

    int64_t out_shape[1] = {cols};
    Tensor* out = tensor_new_cuda(out_shape, 1, grad->dtype);
    if (!out) return NULL;

    // output is zero-initialized by tensor_new_cuda so atomicAdd starts from 0
    int blocks = num_blocks(rows * cols);
    add_broadcast_backward_kernel<<<blocks, BLOCK_SIZE>>>(
        grad->cuda_data, out->cuda_data, rows, cols);
    CUDA_KERNEL_CHECK();
    return out;
}
