#ifndef LUATORCH_NCCL_WRAPPER_H
#define LUATORCH_NCCL_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// initialize nccl across num_gpus devices
// returns 0 on success, -1 on failure
int luatorch_nccl_init(int num_gpus);

// average tensors across all gpus
// ptrs is array of device pointers, one per gpu
// count is number of float elements in each tensor
int luatorch_nccl_allreduce(float** ptrs, int64_t count, int num_gpus);

// fix: broadcast takes per-GPU pointers, not single pointer
int luatorch_nccl_broadcast(float** ptrs, int64_t count, int root, int num_gpus);

// get number of available gpus
int luatorch_nccl_get_gpu_count();

// cleanup
void luatorch_nccl_destroy();

#ifdef __cplusplus
}
#endif

#endif
