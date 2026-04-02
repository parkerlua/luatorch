#include "../tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// matrix multiplication
// this is the most important operation in all of AI
// every layer, every attention calculation, its all matmul

// basic matmul
// a is shape [M, K]
// b is shape [K, N]
// output is shape [M, N]
// every output element is a dot product of a row from a and a column from b
Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;

    // matmul only works on 2D tensors for now
    if (a->ndim != 2 || b->ndim != 2) {
        fprintf(stderr, "luatorch error: matmul requires 2D tensors\n");
        return NULL;
    }

    int64_t M = a->shape[0];  // rows of a
    int64_t K = a->shape[1];  // cols of a, must match rows of b
    int64_t N = b->shape[1];  // cols of b

    // inner dimensions must match
    if (a->shape[1] != b->shape[0]) {
        fprintf(stderr,
            "luatorch error: matmul shape mismatch, got [%lld, %lld] x [%lld, %lld]\n",
            (long long)M, (long long)K,
            (long long)b->shape[0], (long long)N);
        return NULL;
    }

    // output shape is [M, N]
    int64_t out_shape[2] = {M, N};
    Tensor* out = tensor_new(out_shape, 2, a->dtype, a->device);
    if (!out) return NULL;

    // the triple loop
    // i iterates rows of a
    // j iterates cols of b
    // k iterates the shared inner dimension
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                // a[i][k] = a->data[i*K + k]
                // b[k][j] = b->data[k*N + j]
                sum += a->data[i * K + k] * b->data[k * N + j];
            }
            // out[i][j] = sum
            out->data[i * N + j] = sum;
        }
    }

    return out;
}

// batched matmul
// a is shape [batch, M, K]
// b is shape [batch, K, N]
// output is shape [batch, M, N]
// runs matmul independently for each item in the batch
Tensor* tensor_bmm(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;

    if (a->ndim != 3 || b->ndim != 3) {
        fprintf(stderr, "luatorch error: bmm requires 3D tensors\n");
        return NULL;
    }

    int64_t batch = a->shape[0];
    int64_t M     = a->shape[1];
    int64_t K     = a->shape[2];
    int64_t N     = b->shape[2];

    if (a->shape[0] != b->shape[0]) {
        fprintf(stderr, "luatorch error: bmm batch size mismatch\n");
        return NULL;
    }

    if (a->shape[2] != b->shape[1]) {
        fprintf(stderr, "luatorch error: bmm inner dimension mismatch\n");
        return NULL;
    }

    int64_t out_shape[3] = {batch, M, N};
    Tensor* out = tensor_new(out_shape, 3, a->dtype, a->device);
    if (!out) return NULL;

    for (int64_t b_idx = 0; b_idx < batch; b_idx++) {
        // offset into each batch item
        float* a_ptr   = a->data + b_idx * M * K;
        float* b_ptr   = b->data + b_idx * K * N;
        float* out_ptr = out->data + b_idx * M * N;

        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; k++) {
                    sum += a_ptr[i * K + k] * b_ptr[k * N + j];
                }
                out_ptr[i * N + j] = sum;
            }
        }
    }

    return out;
}

// transpose a 2D tensor
// flips rows and cols
// [3, 4] becomes [4, 3]
Tensor* tensor_transpose(Tensor* a) {
    if (!a) return NULL;

    if (a->ndim != 2) {
        fprintf(stderr, "luatorch error: transpose requires 2D tensor\n");
        return NULL;
    }

    int64_t rows = a->shape[0];
    int64_t cols = a->shape[1];

    int64_t out_shape[2] = {cols, rows};
    Tensor* out = tensor_new(out_shape, 2, a->dtype, a->device);
    if (!out) return NULL;

    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            // a[i][j] goes to out[j][i]
            out->data[j * rows + i] = a->data[i * cols + j];
        }
    }

    return out;
}

// dot product of two 1D tensors
// [1,2,3] dot [4,5,6] = 1*4 + 2*5 + 3*6 = 32
float tensor_dot(Tensor* a, Tensor* b) {
    if (!a || !b) return 0.0f;

    if (a->numel != b->numel) {
        fprintf(stderr, "luatorch error: dot product size mismatch\n");
        return 0.0f;
    }

    float sum = 0.0f;
    for (int64_t i = 0; i < a->numel; i++) {
        sum += a->data[i] * b->data[i];
    }
    return sum;
}