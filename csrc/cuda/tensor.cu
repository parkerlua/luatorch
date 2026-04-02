#include "../tensor.h"
#include "memory_pool.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "luatorch cuda error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        return NULL; \
    } \
} while(0)

// move a cpu tensor to the gpu
extern "C" Tensor* tensor_to_cuda(Tensor* t) {
    if (!t) return NULL;
    if (t->device == DEVICE_CUDA) return t;

    size_t bytes = t->numel * sizeof(float);

    float* gpu_ptr = (float*)pool_alloc(bytes);
    if (!gpu_ptr) return NULL;

    cudaError_t err = cudaMemcpy(gpu_ptr, t->data, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "luatorch cuda error: %s\n", cudaGetErrorString(err));
        pool_free(gpu_ptr);
        return NULL;
    }

    free(t->data);
    t->data      = NULL;
    t->cuda_data = gpu_ptr;
    t->device    = DEVICE_CUDA;

    return t;
}

// move a gpu tensor back to cpu
extern "C" Tensor* tensor_to_cpu(Tensor* t) {
    if (!t) return NULL;
    if (t->device == DEVICE_CPU) return t;

    size_t bytes = t->numel * sizeof(float);

    float* cpu_ptr = (float*)malloc(bytes);
    if (!cpu_ptr) {
        fprintf(stderr, "luatorch error: failed to allocate cpu memory for device transfer\n");
        return NULL;
    }

    cudaError_t err = cudaMemcpy(cpu_ptr, t->cuda_data, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "luatorch cuda error: %s\n", cudaGetErrorString(err));
        free(cpu_ptr);
        return NULL;
    }

    pool_free(t->cuda_data);
    t->cuda_data = NULL;
    t->data      = cpu_ptr;
    t->device    = DEVICE_CPU;

    return t;
}

// allocate a new tensor directly on the gpu using the memory pool
extern "C" Tensor* tensor_new_cuda(int64_t* shape, int ndim, DType dtype) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) {
        fprintf(stderr, "luatorch error: failed to allocate tensor struct\n");
        return NULL;
    }

    t->shape = (int64_t*)malloc(ndim * sizeof(int64_t));
    if (!t->shape) { free(t); return NULL; }
    memcpy(t->shape, shape, ndim * sizeof(int64_t));

    t->strides = (int64_t*)malloc(ndim * sizeof(int64_t));
    if (!t->strides) { free(t->shape); free(t); return NULL; }
    t->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * shape[i + 1];
    }

    int64_t numel = 1;
    for (int i = 0; i < ndim; i++) numel *= shape[i];

    t->ndim          = ndim;
    t->numel         = numel;
    t->dtype         = dtype;
    t->device        = DEVICE_CUDA;
    t->requires_grad = 0;
    t->grad          = NULL;
    t->data          = NULL;
    t->ref_count     = 1;

    size_t bytes = numel * sizeof(float);
    float* gpu_ptr = (float*)pool_alloc(bytes);
    if (!gpu_ptr) {
        free(t->strides);
        free(t->shape);
        free(t);
        return NULL;
    }

    cudaMemset(gpu_ptr, 0, bytes);
    t->cuda_data = gpu_ptr;

    return t;
}

// return gpu memory to the pool
extern "C" void tensor_cuda_free(Tensor* t) {
    if (!t) return;
    if (t->cuda_data) {
        pool_free(t->cuda_data);
        t->cuda_data = NULL;
    }
}

// perf fix: async transfer using cuda streams
// allows data loading to overlap with compute
static cudaStream_t transfer_stream = NULL;

static cudaStream_t get_transfer_stream() {
    if (!transfer_stream) {
        cudaStreamCreate(&transfer_stream);
    }
    return transfer_stream;
}

extern "C" Tensor* tensor_to_cuda_async(Tensor* t) {
    if (!t) return NULL;
    if (t->device == DEVICE_CUDA) return t;

    size_t bytes = t->numel * sizeof(float);
    float* gpu_ptr = (float*)pool_alloc(bytes);
    if (!gpu_ptr) return NULL;

    cudaStream_t stream = get_transfer_stream();
    cudaError_t err = cudaMemcpyAsync(gpu_ptr, t->data, bytes,
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "luatorch cuda error: async H2D failed: %s\n", cudaGetErrorString(err));
        pool_free(gpu_ptr);
        return NULL;
    }

    free(t->data);
    t->data      = NULL;
    t->cuda_data = gpu_ptr;
    t->device    = DEVICE_CUDA;
    return t;
}

extern "C" Tensor* tensor_to_cpu_async(Tensor* t) {
    if (!t) return NULL;
    if (t->device == DEVICE_CPU) return t;

    size_t bytes = t->numel * sizeof(float);
    float* cpu_ptr = (float*)malloc(bytes);
    if (!cpu_ptr) return NULL;

    cudaStream_t stream = get_transfer_stream();
    cudaError_t err = cudaMemcpyAsync(cpu_ptr, t->cuda_data, bytes,
        cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "luatorch cuda error: async D2H failed: %s\n", cudaGetErrorString(err));
        free(cpu_ptr);
        return NULL;
    }

    pool_free(t->cuda_data);
    t->cuda_data = NULL;
    t->data      = cpu_ptr;
    t->device    = DEVICE_CPU;
    return t;
}

extern "C" void tensor_sync() {
    if (transfer_stream) {
        cudaStreamSynchronize(transfer_stream);
    }
}

// query gpu memory usage
extern "C" size_t cuda_memory_allocated() {
    return pool_allocated_bytes();
}

extern "C" size_t cuda_memory_cached() {
    return pool_cached_bytes();
}
