#include "../tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// helper to allocate a result tensor with same shape
static Tensor* alloc_result(Tensor* a) {
    return tensor_new(a->shape, a->ndim, a->dtype, a->device);
}

// relu: max(0, x)
// the simplest activation, kills all negative values
Tensor* tensor_relu(Tensor* a) {
    if (!a) return NULL;
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        out->data[i] = a->data[i] > 0.0f ? a->data[i] : 0.0f;
    }
    return out;
}

// sigmoid: 1 / (1 + exp(-x))
// squashes everything into 0 to 1 range
// used for binary classification and gates in LSTMs
Tensor* tensor_sigmoid(Tensor* a) {
    if (!a) return NULL;
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        out->data[i] = 1.0f / (1.0f + expf(-a->data[i]));
    }
    return out;
}

// tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
// squashes into -1 to 1 range
// like sigmoid but centered at zero
Tensor* tensor_tanh(Tensor* a) {
    if (!a) return NULL;
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        out->data[i] = tanhf(a->data[i]);
    }
    return out;
}

// gelu: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// used in transformers, smoother than relu
// approximation formula from the original paper
Tensor* tensor_gelu(Tensor* a) {
    if (!a) return NULL;
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    float sqrt_2_over_pi = 0.7978845608f;
    for (int64_t i = 0; i < a->numel; i++) {
        float x = a->data[i];
        float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
        out->data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
    return out;
}

// silu: x * sigmoid(x)
// also called swish, used in modern architectures like llama
Tensor* tensor_silu(Tensor* a) {
    if (!a) return NULL;
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        float x = a->data[i];
        out->data[i] = x / (1.0f + expf(-x));
    }
    return out;
}

// softmax: exp(x_i) / sum(exp(x_j))
// converts raw scores into probabilities that sum to 1
// subtracts max for numerical stability so exp doesnt overflow
Tensor* tensor_softmax(Tensor* a) {
    if (!a) return NULL;
    Tensor* out = alloc_result(a);
    if (!out) return NULL;

    // for 1D, softmax over all elements
    if (a->ndim == 1) {
        float max_val = a->data[0];
        for (int64_t i = 1; i < a->numel; i++) {
            if (a->data[i] > max_val) max_val = a->data[i];
        }
        float sum = 0.0f;
        for (int64_t i = 0; i < a->numel; i++) {
            out->data[i] = expf(a->data[i] - max_val);
            sum += out->data[i];
        }
        for (int64_t i = 0; i < a->numel; i++) {
            out->data[i] /= sum;
        }
        return out;
    }

    // for 2D, softmax along last dimension (each row)
    if (a->ndim == 2) {
        int64_t rows = a->shape[0];
        int64_t cols = a->shape[1];
        for (int64_t r = 0; r < rows; r++) {
            float* row_in  = a->data + r * cols;
            float* row_out = out->data + r * cols;

            float max_val = row_in[0];
            for (int64_t c = 1; c < cols; c++) {
                if (row_in[c] > max_val) max_val = row_in[c];
            }
            float sum = 0.0f;
            for (int64_t c = 0; c < cols; c++) {
                row_out[c] = expf(row_in[c] - max_val);
                sum += row_out[c];
            }
            for (int64_t c = 0; c < cols; c++) {
                row_out[c] /= sum;
            }
        }
        return out;
    }

    fprintf(stderr, "luatorch error: softmax only supports 1D and 2D tensors\n");
    tensor_free(out);
    return NULL;
}

// compare each element against a scalar
// returns 1.0 where true, 0.0 where false
// useful for making masks in backward passes
Tensor* tensor_gt_scalar(Tensor* a, float scalar) {
    if (!a) return NULL;
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        out->data[i] = a->data[i] > scalar ? 1.0f : 0.0f;
    }
    return out;
}
