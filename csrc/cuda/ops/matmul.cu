#include "../../tensor.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

// forward declaration
extern "C" Tensor* tensor_new_cuda(int64_t* shape, int ndim, DType dtype);

#define CUDA_KERNEL_CHECK() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "luatorch cuda kernel error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        return NULL; \
    } \
} while(0)

// global cublas handle, created once and reused
static cublasHandle_t cublas_handle = NULL;

static cublasHandle_t get_cublas_handle() {
    if (!cublas_handle) {
        cublasStatus_t status = cublasCreate(&cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "luatorch error: failed to create cublas handle\n");
            return NULL;
        }
    }
    return cublas_handle;
}

// naive matmul kernel
// one thread computes one output element
// this is for understanding, cublas version below is what you actually use
__global__ void matmul_naive_kernel(float* a, float* b, float* out,
                                     int64_t M, int64_t K, int64_t N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; k++) {
            sum += a[row * K + k] * b[k * N + col];
        }
        out[row * N + col] = sum;
    }
}

// naive matmul using our own kernel
// a is [M, K], b is [K, N], out is [M, N]
extern "C" Tensor* tensor_matmul_cuda_naive(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;
    if (a->ndim != 2 || b->ndim != 2) {
        fprintf(stderr, "luatorch error: cuda matmul requires 2D tensors\n");
        return NULL;
    }

    int64_t M = a->shape[0];
    int64_t K = a->shape[1];
    int64_t N = b->shape[1];

    if (a->shape[1] != b->shape[0]) {
        fprintf(stderr, "luatorch error: cuda matmul shape mismatch\n");
        return NULL;
    }

    int64_t out_shape[2] = {M, N};
    Tensor* out = tensor_new_cuda(out_shape, 2, a->dtype);
    if (!out) return NULL;

    // 16x16 thread blocks
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    matmul_naive_kernel<<<grid, block>>>(a->cuda_data, b->cuda_data, out->cuda_data, M, K, N);
    CUDA_KERNEL_CHECK();

    return out;
}

// cublas matmul
// this is the real deal, uses tensor cores on the 4090
// cublas uses column major layout so we do the transpose trick:
// instead of C = A * B in row major
// we compute C^T = B^T * A^T in column major
// which is the same result in row major memory layout
extern "C" Tensor* tensor_matmul_cuda(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;
    if (a->ndim != 2 || b->ndim != 2) {
        fprintf(stderr, "luatorch error: cuda matmul requires 2D tensors\n");
        return NULL;
    }

    int64_t M = a->shape[0];
    int64_t K = a->shape[1];
    int64_t N = b->shape[1];

    if (a->shape[1] != b->shape[0]) {
        fprintf(stderr, "luatorch error: cuda matmul shape mismatch\n");
        return NULL;
    }

    cublasHandle_t handle = get_cublas_handle();
    if (!handle) return NULL;

    int64_t out_shape[2] = {M, N};
    Tensor* out = tensor_new_cuda(out_shape, 2, a->dtype);
    if (!out) return NULL;

    float alpha = 1.0f;
    float beta  = 0.0f;

    // cublas is column major, our tensors are row major
    // so we swap A and B: C = A*B in row major == C^T = B^T * A^T in col major
    // lda, ldb, ldc are the leading dimensions
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N,    // no transpose on B (its already "transposed" by the swap)
        CUBLAS_OP_N,    // no transpose on A
        N,              // number of rows of op(B) and C in col major = cols in row major
        M,              // number of cols of op(A) and C in col major = rows in row major
        K,              // shared dimension
        &alpha,
        b->cuda_data, N,    // B with leading dim N
        a->cuda_data, K,    // A with leading dim K
        &beta,
        out->cuda_data, N   // C with leading dim N
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "luatorch error: cublas sgemm failed with status %d\n", (int)status);
        return NULL;
    }

    return out;
}

// batched matmul using cublas
// a is [batch, M, K], b is [batch, K, N], out is [batch, M, N]
extern "C" Tensor* tensor_bmm_cuda(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;
    if (a->ndim != 3 || b->ndim != 3) {
        fprintf(stderr, "luatorch error: cuda bmm requires 3D tensors\n");
        return NULL;
    }

    int64_t batch = a->shape[0];
    int64_t M     = a->shape[1];
    int64_t K     = a->shape[2];
    int64_t N     = b->shape[2];

    if (a->shape[0] != b->shape[0] || a->shape[2] != b->shape[1]) {
        fprintf(stderr, "luatorch error: cuda bmm shape mismatch\n");
        return NULL;
    }

    cublasHandle_t handle = get_cublas_handle();
    if (!handle) return NULL;

    int64_t out_shape[3] = {batch, M, N};
    Tensor* out = tensor_new_cuda(out_shape, 3, a->dtype);
    if (!out) return NULL;

    float alpha = 1.0f;
    float beta  = 0.0f;

    // stride between batch items
    long long int stride_a   = M * K;
    long long int stride_b   = K * N;
    long long int stride_out = M * N;

    cublasStatus_t status = cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        b->cuda_data, N, stride_b,
        a->cuda_data, K, stride_a,
        &beta,
        out->cuda_data, N, stride_out,
        batch
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "luatorch error: cublas batched sgemm failed\n");
        return NULL;
    }

    return out;
}

// transpose on gpu
// swap rows and cols
__global__ void transpose_kernel(float* in, float* out, int64_t rows, int64_t cols) {
    int64_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {
        out[j * rows + i] = in[i * cols + j];
    }
}

extern "C" Tensor* tensor_transpose_cuda(Tensor* a) {
    if (!a || a->ndim != 2) {
        fprintf(stderr, "luatorch error: cuda transpose requires 2D tensor\n");
        return NULL;
    }

    int64_t rows = a->shape[0];
    int64_t cols = a->shape[1];

    int64_t out_shape[2] = {cols, rows};
    Tensor* out = tensor_new_cuda(out_shape, 2, a->dtype);
    if (!out) return NULL;

    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);

    transpose_kernel<<<grid, block>>>(a->cuda_data, out->cuda_data, rows, cols);
    CUDA_KERNEL_CHECK();

    return out;
}

// dot product on gpu
// uses cublas for speed
extern "C" float tensor_dot_cuda(Tensor* a, Tensor* b) {
    if (!a || !b || a->numel != b->numel) {
        fprintf(stderr, "luatorch error: cuda dot product size mismatch\n");
        return 0.0f;
    }

    cublasHandle_t handle = get_cublas_handle();
    if (!handle) return 0.0f;

    float result = 0.0f;
    cublasStatus_t status = cublasSdot(
        handle,
        a->numel,
        a->cuda_data, 1,
        b->cuda_data, 1,
        &result
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "luatorch error: cublas dot failed\n");
        return 0.0f;
    }

    return result;
}
