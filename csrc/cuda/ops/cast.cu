#include "../../tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

extern "C" Tensor* tensor_new_cuda(int64_t* shape, int ndim, DType dtype);

#define BLOCK_SIZE 256

static int num_blocks(int64_t n) {
    return (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

// float32 to float16 kernel
__global__ void f32_to_f16_kernel(float* in, __half* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

// float16 to float32 kernel
__global__ void f16_to_f32_kernel(__half* in, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

// cast float32 tensor to float16 on gpu
// allocates new memory for the half data
extern "C" void* tensor_cast_f32_to_f16_cuda(Tensor* t) {
    if (!t || !t->cuda_data) return NULL;

    size_t bytes = t->numel * sizeof(__half);
    __half* h_ptr;
    cudaError_t err = cudaMalloc((void**)&h_ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "luatorch cuda error: failed to alloc fp16 buffer\n");
        return NULL;
    }

    int blocks = num_blocks(t->numel);
    f32_to_f16_kernel<<<blocks, BLOCK_SIZE>>>(t->cuda_data, h_ptr, t->numel);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "luatorch cuda kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(h_ptr);
        return NULL;
    }

    return (void*)h_ptr;
}

// cast float16 buffer back to float32 into an existing tensor
extern "C" void tensor_cast_f16_to_f32_cuda(void* f16_ptr, Tensor* out) {
    if (!f16_ptr || !out || !out->cuda_data) return;

    int blocks = num_blocks(out->numel);
    f16_to_f32_kernel<<<blocks, BLOCK_SIZE>>>((__half*)f16_ptr, out->cuda_data, out->numel);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "luatorch cuda kernel error: %s\n", cudaGetErrorString(err));
    }
}

// check if any element is inf or nan
// used by grad scaler to detect overflow
__global__ void check_inf_nan_kernel(float* data, int* has_inf_nan, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (isinf(data[i]) || isnan(data[i])) {
        *has_inf_nan = 1;
    }
}

extern "C" int tensor_has_inf_nan_cuda(Tensor* t) {
    if (!t || !t->cuda_data) return 0;

    int* d_flag;
    cudaMalloc((void**)&d_flag, sizeof(int));
    cudaMemset(d_flag, 0, sizeof(int));

    int blocks = num_blocks(t->numel);
    check_inf_nan_kernel<<<blocks, BLOCK_SIZE>>>(t->cuda_data, d_flag, t->numel);

    int result = 0;
    cudaMemcpy(&result, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_flag);
    return result;
}

// scale a tensor by a scalar in place
__global__ void scale_kernel(float* data, float scale, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= scale;
}

extern "C" void tensor_scale_cuda(Tensor* t, float scale) {
    if (!t || !t->cuda_data) return;
    int blocks = num_blocks(t->numel);
    scale_kernel<<<blocks, BLOCK_SIZE>>>(t->cuda_data, scale, t->numel);
}
