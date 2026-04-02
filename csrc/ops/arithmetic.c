#include "../tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// internal helpers

static int check_same_size(Tensor* a, Tensor* b, const char* op) {
    if (a->numel != b->numel) {
        fprintf(stderr,
            "luatorch error: size mismatch in %s, got %lld and %lld\n",
            op,
            (long long)a->numel,
            (long long)b->numel);
        return 0;
    }
    return 1;
}

static Tensor* alloc_result(Tensor* a) {
    return tensor_new(a->shape, a->ndim, a->dtype, a->device);
}

// elementwise operations

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (!check_same_size(a, b, "add")) return NULL;
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return out;
}

Tensor* tensor_sub(Tensor* a, Tensor* b) {
    if (!check_same_size(a, b, "sub")) return NULL;
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
    return out;
}

Tensor* tensor_mul(Tensor* a, Tensor* b) {
    if (!check_same_size(a, b, "mul")) return NULL;
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
    return out;
}

Tensor* tensor_div(Tensor* a, Tensor* b) {
    if (!check_same_size(a, b, "div")) return NULL;
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        if (b->data[i] == 0.0f) {
            fprintf(stderr, "luatorch error: division by zero at index %lld\n", (long long)i);
            out->data[i] = 0.0f;
            continue;
        }
        out->data[i] = a->data[i] / b->data[i];
    }
    return out;
}

// scalar operations

Tensor* tensor_add_scalar(Tensor* a, float scalar) {
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        out->data[i] = a->data[i] + scalar;
    }
    return out;
}

Tensor* tensor_mul_scalar(Tensor* a, float scalar) {
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        out->data[i] = a->data[i] * scalar;
    }
    return out;
}

Tensor* tensor_div_scalar(Tensor* a, float scalar) {
    if (scalar == 0.0f) {
        fprintf(stderr, "luatorch error: division by zero scalar\n");
        return NULL;
    }
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        out->data[i] = a->data[i] / scalar;
    }
    return out;
}

Tensor* tensor_pow_scalar(Tensor* a, float exp) {
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        out->data[i] = powf(a->data[i], exp);
    }
    return out;
}

// unary operations

Tensor* tensor_neg(Tensor* a) {
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        out->data[i] = -a->data[i];
    }
    return out;
}

Tensor* tensor_abs(Tensor* a) {
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        out->data[i] = fabsf(a->data[i]);
    }
    return out;
}

Tensor* tensor_sqrt(Tensor* a) {
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        if (a->data[i] < 0.0f) {
            fprintf(stderr, "luatorch error: sqrt of negative number at index %lld\n", (long long)i);
            out->data[i] = 0.0f;
            continue;
        }
        out->data[i] = sqrtf(a->data[i]);
    }
    return out;
}

Tensor* tensor_log(Tensor* a) {
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        if (a->data[i] <= 0.0f) {
            fprintf(stderr, "luatorch error: log of non-positive number at index %lld\n", (long long)i);
            out->data[i] = -INFINITY;
            continue;
        }
        out->data[i] = logf(a->data[i]);
    }
    return out;
}

Tensor* tensor_exp(Tensor* a) {
    Tensor* out = alloc_result(a);
    if (!out) return NULL;
    for (int64_t i = 0; i < a->numel; i++) {
        out->data[i] = expf(a->data[i]);
    }
    return out;
}

// reduction operations

float tensor_sum(Tensor* a) {
    if (!a) return 0.0f;
    float total = 0.0f;
    for (int64_t i = 0; i < a->numel; i++) {
        total += a->data[i];
    }
    return total;
}

float tensor_mean(Tensor* a) {
    if (!a || a->numel == 0) return 0.0f;
    return tensor_sum(a) / (float)a->numel;
}

float tensor_max(Tensor* a) {
    if (!a || a->numel == 0) return 0.0f;
    float max = a->data[0];
    for (int64_t i = 1; i < a->numel; i++) {
        if (a->data[i] > max) max = a->data[i];
    }
    return max;
}

float tensor_min(Tensor* a) {
    if (!a || a->numel == 0) return 0.0f;
    float min = a->data[0];
    for (int64_t i = 1; i < a->numel; i++) {
        if (a->data[i] < min) min = a->data[i];
    }
    return min;
}

// inplace operations

void tensor_add_(Tensor* a, Tensor* b) {
    if (!check_same_size(a, b, "add_")) return;
    for (int64_t i = 0; i < a->numel; i++) {
        a->data[i] += b->data[i];
    }
}

void tensor_sub_(Tensor* a, Tensor* b) {
    if (!check_same_size(a, b, "sub_")) return;
    for (int64_t i = 0; i < a->numel; i++) {
        a->data[i] -= b->data[i];
    }
}

void tensor_mul_scalar_(Tensor* a, float scalar) {
    if (!a) return;
    for (int64_t i = 0; i < a->numel; i++) {
        a->data[i] *= scalar;
    }
}

void tensor_add_scalar_(Tensor* a, float scalar) {
    if (!a) return;
    for (int64_t i = 0; i < a->numel; i++) {
        a->data[i] += scalar;
    }
}