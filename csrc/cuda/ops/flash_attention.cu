#include "../../tensor.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

extern "C" Tensor* tensor_new_cuda(int64_t* shape, int ndim, DType dtype);

// tile sizes tuned for 4090's 96KB shared memory
// each tile needs Bc*d floats for K and Bc*d floats for V
// for d=64: (64*64 + 64*64) * 4 = 32KB, well within 96KB
#define FLASH_BC 64

// forward kernel
// one block per (batch_head, query_row) pair
// uses shared memory for K and V tiles
// uses dynamically sized accumulator in shared memory instead of stack array
// fix: old code used float acc[128] on stack which crashes if head_dim > 128
__global__ void flash_attention_forward_kernel(
    float* Q, float* K, float* V, float* out,
    int seq_len, int d, float scale, int causal,
    int total_bh // batch_heads, used to compute which bh and row this block handles
) {
    // shared memory layout: [FLASH_BC * d] for sK + [FLASH_BC * d] for sV + [d] for acc
    extern __shared__ float smem[];
    float* sK  = smem;
    float* sV  = smem + FLASH_BC * d;
    float* acc = smem + 2 * FLASH_BC * d;  // fix: accumulator in shared mem, supports any head_dim

    // each block handles one (bh, row) pair
    int block_id = blockIdx.x;
    int bh  = block_id / seq_len;
    int row = block_id % seq_len;

    if (bh >= total_bh || row >= seq_len) return;

    float* q_base = Q + bh * seq_len * d;
    float* k_base = K + bh * seq_len * d;
    float* v_base = V + bh * seq_len * d;
    float* o_base = out + bh * seq_len * d;

    float* q_row = q_base + row * d;

    // initialize accumulator to zero
    if (threadIdx.x == 0) {
        for (int i = 0; i < d; i++) acc[i] = 0.0f;
    }
    __syncthreads();

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    int num_tiles = (seq_len + FLASH_BC - 1) / FLASH_BC;

    for (int tile = 0; tile < num_tiles; tile++) {
        int kv_start = tile * FLASH_BC;
        int kv_end   = kv_start + FLASH_BC;
        if (kv_end > seq_len) kv_end = seq_len;
        int tile_len = kv_end - kv_start;

        // cooperative load of K and V tile into shared memory
        for (int j = threadIdx.x; j < tile_len * d; j += blockDim.x) {
            int kv_idx = j / d;
            int dim_idx = j % d;
            sK[kv_idx * d + dim_idx] = k_base[(kv_start + kv_idx) * d + dim_idx];
            sV[kv_idx * d + dim_idx] = v_base[(kv_start + kv_idx) * d + dim_idx];
        }
        __syncthreads();

        // thread 0 computes attention for this row against the tile
        if (threadIdx.x == 0) {
            for (int j = 0; j < tile_len; j++) {
                int kv_pos = kv_start + j;

                // causal mask: skip future positions
                if (causal && kv_pos > row) continue;

                float score = 0.0f;
                for (int dim = 0; dim < d; dim++) {
                    score += q_row[dim] * sK[j * d + dim];
                }
                score *= scale;

                // online softmax update
                float old_max = row_max;
                if (score > row_max) row_max = score;

                float exp_diff = expf(old_max - row_max);
                row_sum *= exp_diff;
                for (int dim = 0; dim < d; dim++) {
                    acc[dim] *= exp_diff;
                }

                float exp_score = expf(score - row_max);
                row_sum += exp_score;
                for (int dim = 0; dim < d; dim++) {
                    acc[dim] += exp_score * sV[j * d + dim];
                }
            }
        }

        __syncthreads();
    }

    // normalize and write output
    // fix: guard against row_sum == 0 (happens with causal mask on first token edge cases)
    if (threadIdx.x == 0) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        for (int dim = 0; dim < d; dim++) {
            o_base[row * d + dim] = acc[dim] * inv_sum;
        }
    }
}

// flash attention forward
// Q, K, V are [batch_heads * seq_len, head_dim] flattened
// fix: launches all batch_heads * seq_len blocks in one kernel call instead of serial loop
extern "C" Tensor* tensor_flash_attention_cuda(
    Tensor* Q, Tensor* K, Tensor* V,
    int batch_heads, int seq_len, int head_dim, int causal
) {
    if (!Q || !K || !V) return NULL;

    Tensor* out = tensor_new_cuda(Q->shape, Q->ndim, Q->dtype);
    if (!out) return NULL;

    float scale = 1.0f / sqrtf((float)head_dim);

    // shared memory: K tile + V tile + accumulator
    size_t smem_size = (2 * FLASH_BC * head_dim + head_dim) * sizeof(float);

    // fix: single kernel launch for all batch_heads * seq_len rows
    int total_blocks = batch_heads * seq_len;
    flash_attention_forward_kernel<<<total_blocks, 32, smem_size>>>(
        Q->cuda_data, K->cuda_data, V->cuda_data, out->cuda_data,
        seq_len, head_dim, scale, causal, batch_heads);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "luatorch cuda error in flash attention: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    return out;
}
