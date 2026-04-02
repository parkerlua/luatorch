#ifndef LUATORCH_TENSOR_H
#define LUATORCH_TENSOR_H

#include <stdint.h>
#include <stdlib.h>

// what device the tensor lives on
typedef enum {
    DEVICE_CPU = 0,
    DEVICE_CUDA = 1
} Device;

// what type of numbers are stored
typedef enum {
    DTYPE_FLOAT32 = 0,
    DTYPE_FLOAT64 = 1,
    DTYPE_INT32   = 2,
    DTYPE_INT64   = 3,
    DTYPE_FLOAT16 = 4
} DType;

// the actual tensor
typedef struct {
    float*   data;        // cpu data pointer, null if on gpu only
    float*   cuda_data;   // gpu data pointer, null if on cpu only
    int64_t* shape;       // dimensions e.g. [3, 4] for 3x4 matrix
    int64_t* strides;     // how many elements to skip per dimension
    int      ndim;        // number of dimensions
    int64_t  numel;       // total number of elements
    DType    dtype;       // float32, int32 etc
    Device   device;      // cpu or cuda
    int      requires_grad; // should autograd track this
    void*    grad;        // pointer to gradient tensor, null if none
    int      ref_count;   // memory management, free when hits 0
} Tensor;

// create and destroy
Tensor* tensor_new(int64_t* shape, int ndim, DType dtype, Device device);
void    tensor_free(Tensor* t);

// basic info
int64_t tensor_numel(Tensor* t);
void    tensor_print(Tensor* t);

// fill with values
void tensor_fill(Tensor* t, float value);
void tensor_zeros(Tensor* t);
void tensor_ones(Tensor* t);
void tensor_rand(Tensor* t);  // random 0-1
void tensor_randn(Tensor* t); // random normal

// element access
float tensor_get(Tensor* t, int64_t idx);
void  tensor_set(Tensor* t, int64_t idx, float value);

// memory
Tensor* tensor_copy(Tensor* t);
void    tensor_copy_data(Tensor* dst, Tensor* src);
Tensor* tensor_reshape(Tensor* src, int64_t* new_shape, int new_ndim);

// device transfer
Tensor* tensor_to_cuda(Tensor* t);
Tensor* tensor_to_cpu(Tensor* t);

#endif
