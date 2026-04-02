#include "../../tensor.h"
#include <cuda_runtime.h>
#include <stdio.h>

// forward declarations
extern "C" Tensor* tensor_new_cuda(int64_t* shape, int ndim, DType dtype);
extern "C" float tensor_sum_cuda(Tensor* a);
extern "C" void tensor_cuda_free(Tensor* t);

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

// mse kernels

__global__ void mse_diff_sq_kernel(float* pred, float* target, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float diff = pred[i] - target[i];
        out[i] = diff * diff;
    }
}

__global__ void mse_grad_kernel(float* pred, float* target, float* grad, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad[i] = 2.0f * (pred[i] - target[i]) / (float)n;
    }
}

// mae kernels

__global__ void mae_abs_diff_kernel(float* pred, float* target, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fabsf(pred[i] - target[i]);
    }
}

__global__ void mae_grad_kernel(float* pred, float* target, float* grad, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float diff = pred[i] - target[i];
        float scale = 1.0f / (float)n;
        if (diff > 0.0f) grad[i] = scale;
        else if (diff < 0.0f) grad[i] = -scale;
        else grad[i] = 0.0f;
    }
}

// cross entropy kernel
__global__ void cross_entropy_kernel(float* pred, float* target, float* losses,
                                      int64_t batch, int64_t classes) {
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;

    float* row = pred + b * classes;
    int64_t label = (int64_t)target[b];

    // fix: bounds check label to prevent out of bounds access
    if (label < 0 || label >= classes) {
        losses[b] = 0.0f;
        return;
    }

    float max_val = row[0];
    for (int64_t c = 1; c < classes; c++) {
        if (row[c] > max_val) max_val = row[c];
    }

    float lse = 0.0f;
    for (int64_t c = 0; c < classes; c++) {
        lse += expf(row[c] - max_val);
    }
    lse = logf(lse) + max_val;

    losses[b] = -(row[label] - lse);
}

// cross entropy backward kernel
__global__ void cross_entropy_grad_kernel(float* pred, float* target, float* grad,
                                           int64_t batch, int64_t classes) {
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;

    float* row      = pred + b * classes;
    float* grad_row = grad + b * classes;
    int64_t label   = (int64_t)target[b];

    // fix: bounds check label
    if (label < 0 || label >= classes) {
        for (int64_t c = 0; c < classes; c++) grad_row[c] = 0.0f;
        return;
    }

    float max_val = row[0];
    for (int64_t c = 1; c < classes; c++) {
        if (row[c] > max_val) max_val = row[c];
    }
    float sum = 0.0f;
    for (int64_t c = 0; c < classes; c++) {
        grad_row[c] = expf(row[c] - max_val);
        sum += grad_row[c];
    }
    for (int64_t c = 0; c < classes; c++) {
        grad_row[c] /= sum;
    }

    grad_row[label] -= 1.0f;

    for (int64_t c = 0; c < classes; c++) {
        grad_row[c] /= (float)batch;
    }
}

// helper: free a temp tensor through pool_free properly
static void free_temp_tensor(Tensor* t) {
    if (!t) return;
    // fix: use tensor_cuda_free which calls pool_free instead of raw cudaFree
    tensor_cuda_free(t);
    free(t->strides);
    free(t->shape);
    free(t);
}

// cuda loss wrappers

extern "C" Tensor* tensor_mse_loss_cuda(Tensor* pred, Tensor* target) {
    if (!pred || !target || pred->numel != target->numel) return NULL;

    Tensor* diff_sq = tensor_new_cuda(pred->shape, pred->ndim, pred->dtype);
    if (!diff_sq) return NULL;

    int blocks = num_blocks(pred->numel);
    mse_diff_sq_kernel<<<blocks, BLOCK_SIZE>>>(pred->cuda_data, target->cuda_data,
                                                diff_sq->cuda_data, pred->numel);
    CUDA_KERNEL_CHECK();

    float total = tensor_sum_cuda(diff_sq);
    free_temp_tensor(diff_sq);

    int64_t out_shape[1] = {1};
    Tensor* out = tensor_new_cuda(out_shape, 1, pred->dtype);
    if (!out) return NULL;

    float result = total / (float)pred->numel;
    cudaMemcpy(out->cuda_data, &result, sizeof(float), cudaMemcpyHostToDevice);
    return out;
}

extern "C" Tensor* tensor_mse_loss_backward_cuda(Tensor* pred, Tensor* target) {
    if (!pred || !target) return NULL;

    Tensor* grad = tensor_new_cuda(pred->shape, pred->ndim, pred->dtype);
    if (!grad) return NULL;

    int blocks = num_blocks(pred->numel);
    mse_grad_kernel<<<blocks, BLOCK_SIZE>>>(pred->cuda_data, target->cuda_data,
                                             grad->cuda_data, pred->numel);
    CUDA_KERNEL_CHECK();
    return grad;
}

extern "C" Tensor* tensor_mae_loss_cuda(Tensor* pred, Tensor* target) {
    if (!pred || !target || pred->numel != target->numel) return NULL;

    Tensor* abs_diff = tensor_new_cuda(pred->shape, pred->ndim, pred->dtype);
    if (!abs_diff) return NULL;

    int blocks = num_blocks(pred->numel);
    mae_abs_diff_kernel<<<blocks, BLOCK_SIZE>>>(pred->cuda_data, target->cuda_data,
                                                 abs_diff->cuda_data, pred->numel);
    CUDA_KERNEL_CHECK();

    float total = tensor_sum_cuda(abs_diff);
    free_temp_tensor(abs_diff);

    int64_t out_shape[1] = {1};
    Tensor* out = tensor_new_cuda(out_shape, 1, pred->dtype);
    if (!out) return NULL;

    float result = total / (float)pred->numel;
    cudaMemcpy(out->cuda_data, &result, sizeof(float), cudaMemcpyHostToDevice);
    return out;
}

extern "C" Tensor* tensor_mae_loss_backward_cuda(Tensor* pred, Tensor* target) {
    if (!pred || !target) return NULL;

    Tensor* grad = tensor_new_cuda(pred->shape, pred->ndim, pred->dtype);
    if (!grad) return NULL;

    int blocks = num_blocks(pred->numel);
    mae_grad_kernel<<<blocks, BLOCK_SIZE>>>(pred->cuda_data, target->cuda_data,
                                             grad->cuda_data, pred->numel);
    CUDA_KERNEL_CHECK();
    return grad;
}

extern "C" Tensor* tensor_cross_entropy_loss_cuda(Tensor* pred, Tensor* target) {
    if (!pred || !target) return NULL;
    if (pred->ndim != 2 || target->ndim != 1) return NULL;

    int64_t batch   = pred->shape[0];
    int64_t classes = pred->shape[1];

    int64_t losses_shape[1] = {batch};
    Tensor* losses = tensor_new_cuda(losses_shape, 1, pred->dtype);
    if (!losses) return NULL;

    int blocks = num_blocks(batch);
    cross_entropy_kernel<<<blocks, BLOCK_SIZE>>>(pred->cuda_data, target->cuda_data,
                                                  losses->cuda_data, batch, classes);
    CUDA_KERNEL_CHECK();

    float total = tensor_sum_cuda(losses);
    free_temp_tensor(losses);

    int64_t out_shape[1] = {1};
    Tensor* out = tensor_new_cuda(out_shape, 1, pred->dtype);
    if (!out) return NULL;

    float result = total / (float)batch;
    cudaMemcpy(out->cuda_data, &result, sizeof(float), cudaMemcpyHostToDevice);
    return out;
}

extern "C" Tensor* tensor_cross_entropy_loss_backward_cuda(Tensor* pred, Tensor* target) {
    if (!pred || !target) return NULL;

    int64_t batch   = pred->shape[0];
    int64_t classes = pred->shape[1];

    Tensor* grad = tensor_new_cuda(pred->shape, pred->ndim, pred->dtype);
    if (!grad) return NULL;

    int blocks = num_blocks(batch);
    cross_entropy_grad_kernel<<<blocks, BLOCK_SIZE>>>(pred->cuda_data, target->cuda_data,
                                                       grad->cuda_data, batch, classes);
    CUDA_KERNEL_CHECK();
    return grad;
}
