#include "../../tensor.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

extern "C" Tensor* tensor_new_cuda(int64_t* shape, int ndim, DType dtype);

// flash attention 2
// avoids materializing the full N*N attention matrix
// processes attention in tiles using shared memory
// the 4090 has 96KB shared memory per SM
// we use tiles of size Br x Bc where Br, Bc fit in shared memory

// tile sizes tuned for 4090
// each tile needs Br*d + Bc*d floats in shared memory
// for d=64: (64*64 + 64*64) * 4 = 32KB, well within 96KB
#define FLASH_BR 64
#define FLASH_BC 64

// forward kernel
// processes one batch*head at a time
// Q is [seq, d], K is [seq, d], V is [seq, d]
// out is [seq, d]
__global__ void flash_attention_forward_kernel(
    float* Q, float* K, float* V, float* out,
    int seq_len, int d, float scale, int causal
) {
    // shared memory for tiles of K and V
    extern __shared__ float smem[];
    float* sK = smem;                       // [FLASH_BC, d]
    float* sV = smem + FLASH_BC * d;        // [FLASH_BC, d]

    int row = blockIdx.x;  // which query row this block handles
    if (row >= seq_len) return;

    // each row computes its own attention output
    float* q_row = Q + row * d;

    // running softmax: keep track of max and sum of exp
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    // accumulator for output, stored in registers
    // d is typically 64 or 128, fits in registers
    float acc[128];
    for (int i = 0; i < d; i++) acc[i] = 0.0f;

    // iterate over key/value tiles
    int num_tiles = (seq_len + FLASH_BC - 1) / FLASH_BC;

    for (int tile = 0; tile < num_tiles; tile++) {
        int kv_start = tile * FLASH_BC;
        int kv_end   = kv_start + FLASH_BC;
        if (kv_end > seq_len) kv_end = seq_len;
        int tile_len = kv_end - kv_start;

        // load K and V tile into shared memory
        // use threadIdx.x to load cooperatively
        for (int j = threadIdx.x; j < tile_len * d; j += blockDim.x) {
            int kv_idx = j / d;
            int dim_idx = j % d;
            sK[kv_idx * d + dim_idx] = K[(kv_start + kv_idx) * d + dim_idx];
            sV[kv_idx * d + dim_idx] = V[(kv_start + kv_idx) * d + dim_idx];
        }
        __syncthreads();

        // compute attention scores for this tile
        // only thread 0 does the actual computation for this row
        if (threadIdx.x == 0) {
            for (int j = 0; j < tile_len; j++) {
                int kv_pos = kv_start + j;

                // causal mask: skip if key position > query position
                if (causal && kv_pos > row) continue;

                // dot product of q_row and K[j]
                float score = 0.0f;
                for (int dim = 0; dim < d; dim++) {
                    score += q_row[dim] * sK[j * d + dim];
                }
                score *= scale;

                // online softmax update
                float old_max = row_max;
                if (score > row_max) row_max = score;

                // rescale old accumulator
                float exp_diff = expf(old_max - row_max);
                row_sum *= exp_diff;
                for (int dim = 0; dim < d; dim++) {
                    acc[dim] *= exp_diff;
                }

                // add new value
                float exp_score = expf(score - row_max);
                row_sum += exp_score;
                for (int dim = 0; dim < d; dim++) {
                    acc[dim] += exp_score * sV[j * d + dim];
                }
            }
        }

        __syncthreads();
    }

    // normalize by sum and write output
    if (threadIdx.x == 0) {
        float inv_sum = 1.0f / row_sum;
        for (int dim = 0; dim < d; dim++) {
            out[row * d + dim] = acc[dim] * inv_sum;
        }
    }
}

// flash attention forward
// Q, K, V are [batch * heads, seq_len, d] stored as flat [batch*heads*seq_len, d]
// returns output of same shape
extern "C" Tensor* tensor_flash_attention_cuda(
    Tensor* Q, Tensor* K, Tensor* V,
    int batch_heads, int seq_len, int head_dim, int causal
) {
    if (!Q || !K || !V) return NULL;

    // output same shape as Q
    Tensor* out = tensor_new_cuda(Q->shape, Q->ndim, Q->dtype);
    if (!out) return NULL;

    float scale = 1.0f / sqrtf((float)head_dim);

    // shared memory: K tile + V tile
    size_t smem_size = 2 * FLASH_BC * head_dim * sizeof(float);

    // one block per query row per batch*head
    for (int bh = 0; bh < batch_heads; bh++) {
        float* q_ptr = Q->cuda_data + bh * seq_len * head_dim;
        float* k_ptr = K->cuda_data + bh * seq_len * head_dim;
        float* v_ptr = V->cuda_data + bh * seq_len * head_dim;
        float* o_ptr = out->cuda_data + bh * seq_len * head_dim;

        // one block per query position, 32 threads for cooperative loading
        flash_attention_forward_kernel<<<seq_len, 32, smem_size>>>(
            q_ptr, k_ptr, v_ptr, o_ptr,
            seq_len, head_dim, scale, causal);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "luatorch cuda error in flash attention: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    return out;
}
