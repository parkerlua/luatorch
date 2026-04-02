#include "../../tensor.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

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

// fused gelu + bias add in one kernel
// out = gelu(x + bias)
// saves a kernel launch and a memory read/write
__global__ void fused_gelu_bias_kernel(float* x, float* bias, float* out,
                                        int64_t rows, int64_t cols) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    int64_t col = idx % cols;
    float val = x[idx] + bias[col];
    float inner = 0.7978845608f * (val + 0.044715f * val * val * val);
    out[idx] = 0.5f * val * (1.0f + tanhf(inner));
}

extern "C" Tensor* tensor_fused_gelu_bias_cuda(Tensor* x, Tensor* bias) {
    if (!x || !bias) return NULL;

    Tensor* out = tensor_new_cuda(x->shape, x->ndim, x->dtype);
    if (!out) return NULL;

    int64_t rows = x->numel / bias->numel;
    int64_t cols = bias->numel;

    int blocks = num_blocks(x->numel);
    fused_gelu_bias_kernel<<<blocks, BLOCK_SIZE>>>(
        x->cuda_data, bias->cuda_data, out->cuda_data, rows, cols);
    CUDA_KERNEL_CHECK();

    return out;
}

// fused layer norm in one kernel
// each block handles one row
// uses shared memory for mean and variance computation
__global__ void fused_layernorm_kernel(float* x, float* gamma, float* beta, float* out,
                                        int64_t rows, int64_t cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float* x_row   = x   + row * cols;
    float* out_row = out + row * cols;

    // compute mean
    float sum = 0.0f;
    for (int64_t c = 0; c < cols; c++) {
        sum += x_row[c];
    }
    float mean = sum / (float)cols;

    // compute variance
    float var_sum = 0.0f;
    for (int64_t c = 0; c < cols; c++) {
        float diff = x_row[c] - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / (float)cols;
    float inv_std = rsqrtf(var + eps);

    // normalize and apply scale/shift
    for (int64_t c = 0; c < cols; c++) {
        float x_norm = (x_row[c] - mean) * inv_std;
        out_row[c] = gamma[c] * x_norm + beta[c];
    }
}

extern "C" Tensor* tensor_fused_layernorm_cuda(Tensor* x, Tensor* gamma, Tensor* beta, float eps) {
    if (!x || !gamma || !beta) return NULL;

    Tensor* out = tensor_new_cuda(x->shape, x->ndim, x->dtype);
    if (!out) return NULL;

    int64_t cols = gamma->numel;
    int64_t rows = x->numel / cols;

    fused_layernorm_kernel<<<rows, 1>>>(
        x->cuda_data, gamma->cuda_data, beta->cuda_data, out->cuda_data,
        rows, cols, eps);
    CUDA_KERNEL_CHECK();

    return out;
}

// fused adam update in one kernel
// does the entire adam step for one parameter in a single kernel launch
// instead of 8+ separate kernel launches
__global__ void fused_adam_kernel(float* param, float* grad, float* m, float* v,
                                   float lr, float beta1, float beta2, float eps,
                                   float bc1, float bc2, float weight_decay,
                                   int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i];

    // weight decay
    if (weight_decay > 0.0f) {
        param[i] *= (1.0f - lr * weight_decay);
    }

    // update momentum
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;

    // update velocity
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

    // bias corrected
    float m_hat = m[i] / bc1;
    float v_hat = v[i] / bc2;

    // update parameter
    param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

extern "C" void tensor_fused_adam_cuda(Tensor* param, Tensor* grad, Tensor* m, Tensor* v,
                                        float lr, float beta1, float beta2, float eps,
                                        float bc1, float bc2, float weight_decay) {
    if (!param || !grad || !m || !v) return;

    int blocks = num_blocks(param->numel);
    fused_adam_kernel<<<blocks, BLOCK_SIZE>>>(
        param->cuda_data, grad->cuda_data, m->cuda_data, v->cuda_data,
        lr, beta1, beta2, eps, bc1, bc2, weight_decay, param->numel);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "luatorch cuda kernel error: %s\n", cudaGetErrorString(err));
    }
}

// fused softmax + cross entropy in one pass
// avoids materializing the full softmax output
// one thread per batch item
__global__ void fused_cross_entropy_kernel(float* logits, float* target, float* losses,
                                            int64_t batch, int64_t classes) {
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;

    float* row = logits + b * classes;
    int64_t label = (int64_t)target[b];

    // find max for stability
    float max_val = row[0];
    for (int64_t c = 1; c < classes; c++) {
        if (row[c] > max_val) max_val = row[c];
    }

    // log sum exp
    float sum_exp = 0.0f;
    for (int64_t c = 0; c < classes; c++) {
        sum_exp += expf(row[c] - max_val);
    }
    float log_sum_exp = logf(sum_exp) + max_val;

    losses[b] = -(row[label] - log_sum_exp);
}
