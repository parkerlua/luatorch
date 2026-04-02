#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// multiply all shape dimensions together to get total elements
static int64_t calc_numel(int64_t* shape, int ndim) {
    int64_t n = 1;
    for (int i = 0; i < ndim; i++) {
        n *= shape[i];
    }
    return n;
}

// strides tell you how many elements to jump per dimension
static void calc_strides(int64_t* shape, int64_t* strides, int ndim) {
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

static size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return 4;
        case DTYPE_FLOAT64: return 8;
        case DTYPE_INT32:   return 4;
        case DTYPE_INT64:   return 8;
        case DTYPE_FLOAT16: return 2;
        default:            return 4;
    }
}

// random float between 0 and 1, clamped away from 0 for log safety
static float rand_float() {
    float r = (float)rand() / (float)RAND_MAX;
    return r;
}

static float rand_float_safe() {
    float r;
    do {
        r = (float)rand() / (float)RAND_MAX;
    } while (r <= 1e-7f);
    return r;
}

static int seeded = 0;
static void ensure_seeded() {
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
}

Tensor* tensor_new(int64_t* shape, int ndim, DType dtype, Device device) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) {
        fprintf(stderr, "luatorch error: failed to allocate tensor struct\n");
        return NULL;
    }

    t->shape = (int64_t*)malloc(ndim * sizeof(int64_t));
    if (!t->shape) {
        fprintf(stderr, "luatorch error: failed to allocate shape\n");
        free(t);
        return NULL;
    }
    memcpy(t->shape, shape, ndim * sizeof(int64_t));

    t->strides = (int64_t*)malloc(ndim * sizeof(int64_t));
    if (!t->strides) {
        fprintf(stderr, "luatorch error: failed to allocate strides\n");
        free(t->shape);
        free(t);
        return NULL;
    }
    calc_strides(shape, t->strides, ndim);

    t->ndim          = ndim;
    t->numel         = calc_numel(shape, ndim);
    t->dtype         = dtype;
    t->device        = device;
    t->requires_grad = 0;
    t->grad          = NULL;
    t->cuda_data     = NULL;
    t->ref_count     = 1;

    size_t bytes = t->numel * dtype_size(dtype);
    t->data = (float*)malloc(bytes);
    if (!t->data) {
        fprintf(stderr, "luatorch error: failed to allocate data (%zu bytes)\n", bytes);
        free(t->strides);
        free(t->shape);
        free(t);
        return NULL;
    }

    memset(t->data, 0, bytes);
    return t;
}

// forward declare cuda free, may not be linked if no cuda
extern void tensor_cuda_free(Tensor* t);

void tensor_free(Tensor* t) {
    if (!t) return;

    t->ref_count--;
    if (t->ref_count > 0) return;

    if (t->grad) {
        tensor_free((Tensor*)t->grad);
        t->grad = NULL;
    }

    if (t->cuda_data) {
        tensor_cuda_free(t);
    }

    if (t->data)    free(t->data);
    if (t->strides) free(t->strides);
    if (t->shape)   free(t->shape);

    free(t);
}

void tensor_fill(Tensor* t, float value) {
    if (!t || !t->data) return;
    for (int64_t i = 0; i < t->numel; i++) {
        t->data[i] = value;
    }
}

void tensor_zeros(Tensor* t) {
    if (!t || !t->data) return;
    memset(t->data, 0, t->numel * dtype_size(t->dtype));
}

void tensor_ones(Tensor* t) {
    tensor_fill(t, 1.0f);
}

void tensor_rand(Tensor* t) {
    if (!t || !t->data) return;
    ensure_seeded();
    for (int64_t i = 0; i < t->numel; i++) {
        t->data[i] = rand_float();
    }
}

// Box-Muller transform for normal distribution
// uses rand_float_safe to avoid log(0) which gives -inf
void tensor_randn(Tensor* t) {
    if (!t || !t->data) return;
    ensure_seeded();
    for (int64_t i = 0; i < t->numel - 1; i += 2) {
        float u1  = rand_float_safe();
        float u2  = rand_float();
        float mag = sqrtf(-2.0f * logf(u1));
        t->data[i]     = mag * cosf(2.0f * 3.14159265f * u2);
        t->data[i + 1] = mag * sinf(2.0f * 3.14159265f * u2);
    }
    if (t->numel % 2 != 0) {
        float u1  = rand_float_safe();
        float u2  = rand_float();
        float mag = sqrtf(-2.0f * logf(u1));
        t->data[t->numel - 1] = mag * cosf(2.0f * 3.14159265f * u2);
    }
}

Tensor* tensor_copy(Tensor* src) {
    if (!src) return NULL;

    Tensor* dst = tensor_new(src->shape, src->ndim, src->dtype, src->device);
    if (!dst) return NULL;

    // only copy if data exists (cpu tensor)
    if (src->data && dst->data) {
        memcpy(dst->data, src->data, src->numel * dtype_size(src->dtype));
    }
    dst->requires_grad = src->requires_grad;

    return dst;
}

void tensor_copy_data(Tensor* dst, Tensor* src) {
    if (!dst || !src) return;
    if (dst->numel != src->numel) {
        fprintf(stderr, "luatorch error: tensor size mismatch in copy_data\n");
        return;
    }
    if (src->data && dst->data) {
        memcpy(dst->data, src->data, src->numel * dtype_size(src->dtype));
    }
}

void tensor_print(Tensor* t) {
    if (!t) {
        printf("Tensor(NULL)\n");
        return;
    }

    printf("Tensor(shape=[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%lld", (long long)t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("], dtype=float32, device=%s)\n",
        t->device == DEVICE_CPU ? "cpu" : "cuda");

    // only print data if on cpu
    if (!t->data) {
        printf("[data on gpu, use cpu() to view]\n");
        return;
    }

    printf("[");
    int64_t limit = t->numel < 20 ? t->numel : 20;
    for (int64_t i = 0; i < limit; i++) {
        printf("%.4f", t->data[i]);
        if (i < limit - 1) printf(", ");
    }
    if (t->numel > 20) printf(", ...");
    printf("]\n");
}

int64_t tensor_numel(Tensor* t) {
    if (!t) return 0;
    return t->numel;
}

// get/set check for cuda and warn instead of crashing
float tensor_get(Tensor* t, int64_t idx) {
    if (!t || idx >= t->numel || idx < 0) {
        fprintf(stderr, "luatorch error: index %lld out of bounds\n", (long long)idx);
        return 0.0f;
    }
    if (!t->data) {
        fprintf(stderr, "luatorch error: tensor_get on gpu tensor, call cpu() first\n");
        return 0.0f;
    }
    return t->data[idx];
}

void tensor_set(Tensor* t, int64_t idx, float value) {
    if (!t || idx >= t->numel || idx < 0) {
        fprintf(stderr, "luatorch error: index %lld out of bounds\n", (long long)idx);
        return;
    }
    if (!t->data) {
        fprintf(stderr, "luatorch error: tensor_set on gpu tensor, call cpu() first\n");
        return;
    }
    t->data[idx] = value;
}

Tensor* tensor_reshape(Tensor* src, int64_t* new_shape, int new_ndim) {
    if (!src) return NULL;

    int64_t new_numel = calc_numel(new_shape, new_ndim);
    if (new_numel != src->numel) {
        fprintf(stderr,
            "luatorch error: reshape size mismatch, %lld vs %lld\n",
            (long long)src->numel,
            (long long)new_numel);
        return NULL;
    }

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->data          = src->data;
    t->cuda_data     = src->cuda_data;
    t->numel         = src->numel;
    t->dtype         = src->dtype;
    t->device        = src->device;
    t->ndim          = new_ndim;
    t->requires_grad = src->requires_grad;
    t->grad          = NULL;
    t->ref_count     = 1;

    t->shape = (int64_t*)malloc(new_ndim * sizeof(int64_t));
    memcpy(t->shape, new_shape, new_ndim * sizeof(int64_t));

    t->strides = (int64_t*)malloc(new_ndim * sizeof(int64_t));
    calc_strides(new_shape, t->strides, new_ndim);

    src->ref_count++;

    return t;
}
