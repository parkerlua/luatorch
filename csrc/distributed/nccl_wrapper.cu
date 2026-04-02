#include "nccl_wrapper.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>

static ncclComm_t* comms = NULL;
static cudaStream_t* streams = NULL;
static int initialized = 0;
static int n_gpus = 0;

#define NCCL_CHECK(call) do { \
    ncclResult_t res = (call); \
    if (res != ncclSuccess) { \
        fprintf(stderr, "luatorch nccl error at %s:%d: %s\n", \
            __FILE__, __LINE__, ncclGetErrorString(res)); \
        return -1; \
    } \
} while(0)

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "luatorch cuda error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

extern "C" int luatorch_nccl_get_gpu_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return 0;
    return count;
}

extern "C" int luatorch_nccl_init(int num_gpus) {
    if (initialized) return 0;

    int available = luatorch_nccl_get_gpu_count();
    if (num_gpus > available) {
        fprintf(stderr, "luatorch error: requested %d gpus but only %d available\n",
            num_gpus, available);
        return -1;
    }

    n_gpus = num_gpus;

    // fix: NULL checks on all mallocs, cleanup on failure
    comms = (ncclComm_t*)malloc(num_gpus * sizeof(ncclComm_t));
    if (!comms) {
        fprintf(stderr, "luatorch error: failed to allocate nccl communicators\n");
        return -1;
    }
    streams = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));
    if (!streams) {
        fprintf(stderr, "luatorch error: failed to allocate cuda streams\n");
        free(comms); comms = NULL;
        return -1;
    }

    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    int* devs = (int*)malloc(num_gpus * sizeof(int));
    if (!devs) {
        fprintf(stderr, "luatorch error: failed to allocate device list\n");
        free(streams); streams = NULL;
        free(comms); comms = NULL;
        return -1;
    }
    for (int i = 0; i < num_gpus; i++) devs[i] = i;

    // initialize nccl
    NCCL_CHECK(ncclCommInitAll(comms, num_gpus, devs));

    free(devs);
    initialized = 1;

    printf("luatorch: nccl initialized with %d gpus\n", num_gpus);
    return 0;
}

// average a tensor across all gpus using allreduce
// each gpu has its own copy of the tensor, after this call they all have the average
extern "C" int luatorch_nccl_allreduce(float** ptrs, int64_t count, int num_gpus) {
    if (!initialized) {
        fprintf(stderr, "luatorch error: nccl not initialized\n");
        return -1;
    }

    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        // allreduce in place, average across gpus
        NCCL_CHECK(ncclAllReduce(
            ptrs[i], ptrs[i], count,
            ncclFloat, ncclAvg, comms[i], streams[i]));
    }
    NCCL_CHECK(ncclGroupEnd());

    // sync all streams
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    return 0;
}

// fix: broadcast now takes per-GPU pointers instead of single pointer
// old API was wrong because each GPU has its own address space
// root GPU's data gets copied to all other GPUs' buffers
extern "C" int luatorch_nccl_broadcast(float** ptrs, int64_t count, int root, int num_gpus) {
    if (!initialized) {
        fprintf(stderr, "luatorch error: nccl not initialized\n");
        return -1;
    }

    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        NCCL_CHECK(ncclBroadcast(ptrs[root], ptrs[i], count, ncclFloat, root, comms[i], streams[i]));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    return 0;
}

extern "C" void luatorch_nccl_destroy() {
    if (!initialized) return;

    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamDestroy(streams[i]);
        ncclCommDestroy(comms[i]);
    }

    free(comms);
    free(streams);
    comms = NULL;
    streams = NULL;
    initialized = 0;
}
