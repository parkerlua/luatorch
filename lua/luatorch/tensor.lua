local ffi = require('ffi')

-- tell LuaJIT about the C structs and functions
ffi.cdef[[
    typedef enum { DEVICE_CPU = 0, DEVICE_CUDA = 1 } Device;

    typedef enum {
        DTYPE_FLOAT32 = 0,
        DTYPE_FLOAT64 = 1,
        DTYPE_INT32   = 2,
        DTYPE_INT64   = 3,
        DTYPE_FLOAT16 = 4
    } DType;

    typedef struct {
        float*   data;
        float*   cuda_data;
        int64_t* shape;
        int64_t* strides;
        int      ndim;
        int64_t  numel;
        int      dtype;
        int      device;
        int      requires_grad;
        void*    grad;
        int      ref_count;
    } Tensor;

    // tensor.c
    Tensor* tensor_new(int64_t* shape, int ndim, int dtype, int device);
    void    tensor_free(Tensor* t);
    void    tensor_fill(Tensor* t, float value);
    void    tensor_zeros(Tensor* t);
    void    tensor_ones(Tensor* t);
    void    tensor_rand(Tensor* t);
    void    tensor_randn(Tensor* t);
    void    tensor_print(Tensor* t);
    Tensor* tensor_copy(Tensor* t);
    void    tensor_copy_data(Tensor* dst, Tensor* src);
    float   tensor_get(Tensor* t, int64_t idx);
    void    tensor_set(Tensor* t, int64_t idx, float value);
    Tensor* tensor_reshape(Tensor* src, int64_t* new_shape, int new_ndim);
    int64_t tensor_numel(Tensor* t);

    // ops/arithmetic.c (cpu)
    Tensor* tensor_add(Tensor* a, Tensor* b);
    Tensor* tensor_sub(Tensor* a, Tensor* b);
    Tensor* tensor_mul(Tensor* a, Tensor* b);
    Tensor* tensor_div(Tensor* a, Tensor* b);
    Tensor* tensor_add_scalar(Tensor* a, float scalar);
    Tensor* tensor_mul_scalar(Tensor* a, float scalar);
    Tensor* tensor_div_scalar(Tensor* a, float scalar);
    Tensor* tensor_pow_scalar(Tensor* a, float exp);
    Tensor* tensor_neg(Tensor* a);
    Tensor* tensor_abs(Tensor* a);
    Tensor* tensor_sqrt(Tensor* a);
    Tensor* tensor_log(Tensor* a);
    Tensor* tensor_exp(Tensor* a);
    float   tensor_sum(Tensor* a);
    float   tensor_mean(Tensor* a);
    float   tensor_max(Tensor* a);
    float   tensor_min(Tensor* a);
    void    tensor_add_(Tensor* a, Tensor* b);
    void    tensor_sub_(Tensor* a, Tensor* b);
    void    tensor_mul_scalar_(Tensor* a, float scalar);
    void    tensor_add_scalar_(Tensor* a, float scalar);
    Tensor* tensor_add_broadcast(Tensor* a, Tensor* b);
    Tensor* tensor_add_broadcast_backward(Tensor* grad);

    // ops/conv2d.c
    Tensor* tensor_im2col(Tensor* input, int batch, int channels, int height, int width,
                           int kernel_size, int stride, int padding);

    // ops/matmul.c (cpu)
    Tensor* tensor_matmul(Tensor* a, Tensor* b);
    Tensor* tensor_bmm(Tensor* a, Tensor* b);
    Tensor* tensor_transpose(Tensor* a);
    float   tensor_dot(Tensor* a, Tensor* b);

    // ops/activation.c (cpu)
    Tensor* tensor_relu(Tensor* a);
    Tensor* tensor_sigmoid(Tensor* a);
    Tensor* tensor_tanh(Tensor* a);
    Tensor* tensor_gelu(Tensor* a);
    Tensor* tensor_silu(Tensor* a);
    Tensor* tensor_softmax(Tensor* a);
    Tensor* tensor_gt_scalar(Tensor* a, float scalar);

    // ops/loss.c (cpu)
    Tensor* tensor_mse_loss(Tensor* pred, Tensor* target);
    Tensor* tensor_mse_loss_backward(Tensor* pred, Tensor* target);
    Tensor* tensor_mae_loss(Tensor* pred, Tensor* target);
    Tensor* tensor_mae_loss_backward(Tensor* pred, Tensor* target);
    Tensor* tensor_cross_entropy_loss(Tensor* pred, Tensor* target);
    Tensor* tensor_cross_entropy_loss_backward(Tensor* pred, Tensor* target);

    // cuda/tensor.cu
    Tensor* tensor_to_cuda(Tensor* t);
    Tensor* tensor_to_cpu(Tensor* t);
    Tensor* tensor_to_cuda_async(Tensor* t);
    Tensor* tensor_to_cpu_async(Tensor* t);
    void    tensor_sync();
    Tensor* tensor_new_cuda(int64_t* shape, int ndim, int dtype);
    void    tensor_cuda_free(Tensor* t);

    // cuda/ops/arithmetic.cu
    Tensor* tensor_add_cuda(Tensor* a, Tensor* b);
    Tensor* tensor_sub_cuda(Tensor* a, Tensor* b);
    Tensor* tensor_mul_cuda(Tensor* a, Tensor* b);
    Tensor* tensor_div_cuda(Tensor* a, Tensor* b);
    Tensor* tensor_add_scalar_cuda(Tensor* a, float scalar);
    Tensor* tensor_mul_scalar_cuda(Tensor* a, float scalar);
    Tensor* tensor_div_scalar_cuda(Tensor* a, float scalar);
    Tensor* tensor_pow_scalar_cuda(Tensor* a, float exp);
    Tensor* tensor_neg_cuda(Tensor* a);
    Tensor* tensor_abs_cuda(Tensor* a);
    Tensor* tensor_sqrt_cuda(Tensor* a);
    Tensor* tensor_log_cuda(Tensor* a);
    Tensor* tensor_exp_cuda(Tensor* a);
    float   tensor_sum_cuda(Tensor* a);
    float   tensor_mean_cuda(Tensor* a);
    float   tensor_max_cuda(Tensor* a);
    float   tensor_min_cuda(Tensor* a);
    void    tensor_add_cuda_(Tensor* a, Tensor* b);
    void    tensor_sub_cuda_(Tensor* a, Tensor* b);
    void    tensor_mul_scalar_cuda_(Tensor* a, float scalar);
    void    tensor_add_scalar_cuda_(Tensor* a, float scalar);
    Tensor* tensor_add_broadcast_cuda(Tensor* a, Tensor* b);
    Tensor* tensor_add_broadcast_backward_cuda(Tensor* grad);

    // cuda/ops/matmul.cu
    Tensor* tensor_matmul_cuda(Tensor* a, Tensor* b);
    Tensor* tensor_matmul_cuda_naive(Tensor* a, Tensor* b);
    Tensor* tensor_bmm_cuda(Tensor* a, Tensor* b);
    Tensor* tensor_transpose_cuda(Tensor* a);
    float   tensor_dot_cuda(Tensor* a, Tensor* b);

    // cuda/ops/activation.cu
    Tensor* tensor_relu_cuda(Tensor* a);
    Tensor* tensor_sigmoid_cuda(Tensor* a);
    Tensor* tensor_tanh_cuda(Tensor* a);
    Tensor* tensor_gelu_cuda(Tensor* a);
    Tensor* tensor_silu_cuda(Tensor* a);
    Tensor* tensor_softmax_cuda(Tensor* a);
    Tensor* tensor_gt_scalar_cuda(Tensor* a, float scalar);

    // cuda/ops/loss.cu
    Tensor* tensor_mse_loss_cuda(Tensor* pred, Tensor* target);
    Tensor* tensor_mse_loss_backward_cuda(Tensor* pred, Tensor* target);
    Tensor* tensor_mae_loss_cuda(Tensor* pred, Tensor* target);
    Tensor* tensor_mae_loss_backward_cuda(Tensor* pred, Tensor* target);
    Tensor* tensor_cross_entropy_loss_cuda(Tensor* pred, Tensor* target);
    Tensor* tensor_cross_entropy_loss_backward_cuda(Tensor* pred, Tensor* target);

    // cuda/ops/fused.cu
    Tensor* tensor_fused_gelu_bias_cuda(Tensor* x, Tensor* bias);
    Tensor* tensor_fused_layernorm_cuda(Tensor* x, Tensor* gamma, Tensor* beta, float eps);
    void    tensor_fused_adam_cuda(Tensor* param, Tensor* grad, Tensor* m, Tensor* v,
                                   float lr, float beta1, float beta2, float eps,
                                   float bc1, float bc2, float weight_decay);

    // cuda/ops/flash_attention.cu
    Tensor* tensor_flash_attention_cuda(Tensor* Q, Tensor* K, Tensor* V,
                                         int batch_heads, int seq_len, int head_dim, int causal);

    // cuda/ops/cast.cu
    int  tensor_has_inf_nan_cuda(Tensor* t);
    void tensor_scale_cuda(Tensor* t, float scale);

    // cuda/memory_pool
    void   pool_init();
    void   pool_clear();
    size_t cuda_memory_allocated();
    size_t cuda_memory_cached();
]]

-- load the compiled C library
local lib = ffi.load('luatorch')

-- perf fix: cache all hot FFI calls as locals
-- LuaJIT traces through local FFI calls much better than lib.x lookups
local C_new           = lib.tensor_new
local C_free          = lib.tensor_free
local C_fill          = lib.tensor_fill
local C_zeros         = lib.tensor_zeros
local C_ones          = lib.tensor_ones
local C_rand          = lib.tensor_rand
local C_randn         = lib.tensor_randn
local C_print         = lib.tensor_print
local C_copy          = lib.tensor_copy
local C_get           = lib.tensor_get
local C_set           = lib.tensor_set
local C_add           = lib.tensor_add
local C_sub           = lib.tensor_sub
local C_mul           = lib.tensor_mul
local C_div           = lib.tensor_div
local C_add_scalar    = lib.tensor_add_scalar
local C_mul_scalar    = lib.tensor_mul_scalar
local C_div_scalar    = lib.tensor_div_scalar
local C_pow_scalar    = lib.tensor_pow_scalar
local C_neg           = lib.tensor_neg
local C_abs           = lib.tensor_abs
local C_sqrt          = lib.tensor_sqrt
local C_log           = lib.tensor_log
local C_exp           = lib.tensor_exp
local C_sum           = lib.tensor_sum
local C_mean          = lib.tensor_mean
local C_max           = lib.tensor_max
local C_min           = lib.tensor_min
local C_add_          = lib.tensor_add_
local C_sub_          = lib.tensor_sub_
local C_mul_scalar_   = lib.tensor_mul_scalar_
local C_add_scalar_   = lib.tensor_add_scalar_
local C_add_bc        = lib.tensor_add_broadcast
local C_add_bc_bwd    = lib.tensor_add_broadcast_backward
local C_matmul        = lib.tensor_matmul
local C_bmm           = lib.tensor_bmm
local C_transpose     = lib.tensor_transpose
local C_dot           = lib.tensor_dot
local C_relu          = lib.tensor_relu
local C_sigmoid       = lib.tensor_sigmoid
local C_tanh          = lib.tensor_tanh
local C_gelu          = lib.tensor_gelu
local C_silu          = lib.tensor_silu
local C_softmax       = lib.tensor_softmax
local C_gt_scalar     = lib.tensor_gt_scalar
local C_to_cuda       = lib.tensor_to_cuda
local C_to_cpu        = lib.tensor_to_cpu
-- cuda variants
local C_add_cuda      = lib.tensor_add_cuda
local C_sub_cuda      = lib.tensor_sub_cuda
local C_mul_cuda      = lib.tensor_mul_cuda
local C_div_cuda      = lib.tensor_div_cuda
local C_add_scalar_cuda = lib.tensor_add_scalar_cuda
local C_mul_scalar_cuda = lib.tensor_mul_scalar_cuda
local C_div_scalar_cuda = lib.tensor_div_scalar_cuda
local C_pow_scalar_cuda = lib.tensor_pow_scalar_cuda
local C_neg_cuda      = lib.tensor_neg_cuda
local C_abs_cuda      = lib.tensor_abs_cuda
local C_sqrt_cuda     = lib.tensor_sqrt_cuda
local C_log_cuda      = lib.tensor_log_cuda
local C_exp_cuda      = lib.tensor_exp_cuda
local C_sum_cuda      = lib.tensor_sum_cuda
local C_mean_cuda     = lib.tensor_mean_cuda
local C_max_cuda      = lib.tensor_max_cuda
local C_min_cuda      = lib.tensor_min_cuda
local C_add_cuda_     = lib.tensor_add_cuda_
local C_sub_cuda_     = lib.tensor_sub_cuda_
local C_mul_scalar_cuda_ = lib.tensor_mul_scalar_cuda_
local C_add_scalar_cuda_ = lib.tensor_add_scalar_cuda_
local C_add_bc_cuda   = lib.tensor_add_broadcast_cuda
local C_add_bc_bwd_cuda = lib.tensor_add_broadcast_backward_cuda
local C_matmul_cuda   = lib.tensor_matmul_cuda
local C_bmm_cuda      = lib.tensor_bmm_cuda
local C_transpose_cuda = lib.tensor_transpose_cuda
local C_dot_cuda      = lib.tensor_dot_cuda
local C_relu_cuda     = lib.tensor_relu_cuda
local C_sigmoid_cuda  = lib.tensor_sigmoid_cuda
local C_tanh_cuda     = lib.tensor_tanh_cuda
local C_gelu_cuda     = lib.tensor_gelu_cuda
local C_silu_cuda     = lib.tensor_silu_cuda
local C_softmax_cuda  = lib.tensor_softmax_cuda
local C_gt_scalar_cuda = lib.tensor_gt_scalar_cuda
local C_im2col        = lib.tensor_im2col

-- the Lua facing tensor object
local Tensor = {}
Tensor.__index = Tensor

-- helper to wrap a raw C tensor pointer into a Lua Tensor object
local function wrap_raw(raw, shape, ndim, dtype, device)
    local t = setmetatable({}, Tensor)
    t._raw   = raw
    t.shape  = shape
    t.ndim   = ndim
    t.dtype  = dtype  or 'float32'
    t.device = device or 'cpu'
    return t
end

-- read shape from a raw C tensor pointer
local function read_shape(raw, ndim)
    local shape = {}
    for i = 0, ndim - 1 do
        shape[i + 1] = tonumber(raw.shape[i])
    end
    return shape
end

-- wrap a C result pointer into a Lua tensor
local function wrap_result(raw_result, ref_tensor)
    if raw_result == nil then return nil end
    local ndim  = raw_result.ndim
    local shape = read_shape(raw_result, ndim)
    local device = raw_result.device == 1 and 'cuda' or 'cpu'
    return wrap_raw(raw_result, shape, ndim, ref_tensor.dtype, device)
end

-- check if tensor is on gpu
local function is_cuda(t)
    return t.device == 'cuda'
end

-- constructor
function Tensor.new(shape, dtype, device)
    dtype  = dtype  or 'float32'
    device = device or 'cpu'

    local ndim    = #shape
    local c_shape = ffi.new('int64_t[?]', ndim)
    for i, v in ipairs(shape) do
        c_shape[i-1] = v
    end

    local dtype_map = {
        float32 = 0, float64 = 1, int32 = 2, int64 = 3, float16 = 4
    }
    local device_map = { cpu = 0, cuda = 1 }

    local raw = C_new(c_shape, ndim, dtype_map[dtype], device_map[device])
    return wrap_raw(raw, shape, ndim, dtype, device)
end

-- clean up C memory when Lua object is garbage collected
function Tensor:__gc()
    if self._raw ~= nil then
        C_free(self._raw)
        self._raw = nil
    end
end

-- move tensor to gpu
function Tensor:cuda()
    if is_cuda(self) then return self end
    C_to_cuda(self._raw)
    self.device = 'cuda'
    return self
end

-- move tensor to cpu
function Tensor:cpu()
    if not is_cuda(self) then return self end
    C_to_cpu(self._raw)
    self.device = 'cpu'
    return self
end

-- perf: async device transfer, overlaps data loading with compute
-- call sync() before reading data
function Tensor:cuda_async()
    if is_cuda(self) then return self end
    lib.tensor_to_cuda_async(self._raw)
    self.device = 'cuda'
    return self
end

function Tensor:cpu_async()
    if not is_cuda(self) then return self end
    lib.tensor_to_cpu_async(self._raw)
    self.device = 'cpu'
    return self
end

function Tensor.sync()
    lib.tensor_sync()
end

-- fill operations (cpu only, fill then move if needed)
function Tensor:fill(value)
    C_fill(self._raw, value)
    return self
end

function Tensor:zeros()
    C_zeros(self._raw)
    return self
end

function Tensor:ones()
    C_ones(self._raw)
    return self
end

function Tensor:rand()
    C_rand(self._raw)
    return self
end

function Tensor:randn()
    C_randn(self._raw)
    return self
end

function Tensor:print()
    C_print(self._raw)
    return self
end

function Tensor:copy()
    local raw_copy = C_copy(self._raw)
    return wrap_raw(raw_copy, {table.unpack(self.shape)}, self.ndim, self.dtype, self.device)
end

function Tensor:get(idx)
    return C_get(self._raw, idx)
end

function Tensor:set(idx, value)
    C_set(self._raw, idx, value)
    return self
end

function Tensor:numel()
    local n = 1
    for _, v in ipairs(self.shape) do n = n * v end
    return n
end

function Tensor:__tostring()
    return string.format(
        'Tensor(shape=%s, dtype=%s, device=%s)',
        table.concat(self.shape, 'x'),
        self.dtype,
        self.device
    )
end

-- dispatch: picks cpu or cuda version based on tensor device
-- for binary ops, checks first tensor

-- elementwise ops
function Tensor.add(a, b)
    if is_cuda(a) then return wrap_result(C_add_cuda(a._raw, b._raw), a) end
    return wrap_result(C_add(a._raw, b._raw), a)
end

function Tensor.sub(a, b)
    if is_cuda(a) then return wrap_result(C_sub_cuda(a._raw, b._raw), a) end
    return wrap_result(C_sub(a._raw, b._raw), a)
end

function Tensor.mul(a, b)
    if is_cuda(a) then return wrap_result(C_mul_cuda(a._raw, b._raw), a) end
    return wrap_result(C_mul(a._raw, b._raw), a)
end

function Tensor.div(a, b)
    if is_cuda(a) then return wrap_result(C_div_cuda(a._raw, b._raw), a) end
    return wrap_result(C_div(a._raw, b._raw), a)
end

-- scalar ops
function Tensor.add_scalar(a, s)
    if is_cuda(a) then return wrap_result(C_add_scalar_cuda(a._raw, s), a) end
    return wrap_result(C_add_scalar(a._raw, s), a)
end

function Tensor.mul_scalar(a, s)
    if is_cuda(a) then return wrap_result(C_mul_scalar_cuda(a._raw, s), a) end
    return wrap_result(C_mul_scalar(a._raw, s), a)
end

function Tensor.div_scalar(a, s)
    if is_cuda(a) then return wrap_result(C_div_scalar_cuda(a._raw, s), a) end
    return wrap_result(C_div_scalar(a._raw, s), a)
end

function Tensor.pow_scalar(a, e)
    if is_cuda(a) then return wrap_result(C_pow_scalar_cuda(a._raw, e), a) end
    return wrap_result(C_pow_scalar(a._raw, e), a)
end

-- unary ops
function Tensor.neg(a)
    if is_cuda(a) then return wrap_result(C_neg_cuda(a._raw), a) end
    return wrap_result(C_neg(a._raw), a)
end

function Tensor.abs(a)
    if is_cuda(a) then return wrap_result(C_abs_cuda(a._raw), a) end
    return wrap_result(C_abs(a._raw), a)
end

function Tensor.sqrt(a)
    if is_cuda(a) then return wrap_result(C_sqrt_cuda(a._raw), a) end
    return wrap_result(C_sqrt(a._raw), a)
end

function Tensor.log(a)
    if is_cuda(a) then return wrap_result(C_log_cuda(a._raw), a) end
    return wrap_result(C_log(a._raw), a)
end

function Tensor.exp(a)
    if is_cuda(a) then return wrap_result(C_exp_cuda(a._raw), a) end
    return wrap_result(C_exp(a._raw), a)
end

-- reductions
function Tensor.sum(a)
    if is_cuda(a) then return tonumber(C_sum_cuda(a._raw)) end
    return tonumber(C_sum(a._raw))
end

function Tensor.mean(a)
    if is_cuda(a) then return tonumber(C_mean_cuda(a._raw)) end
    return tonumber(C_mean(a._raw))
end

function Tensor.max(a)
    if is_cuda(a) then return tonumber(C_max_cuda(a._raw)) end
    return tonumber(C_max(a._raw))
end

function Tensor.min(a)
    if is_cuda(a) then return tonumber(C_min_cuda(a._raw)) end
    return tonumber(C_min(a._raw))
end

-- inplace ops
function Tensor.add_(a, b)
    if is_cuda(a) then C_add_cuda_(a._raw, b._raw) else C_add_(a._raw, b._raw) end
end

function Tensor.sub_(a, b)
    if is_cuda(a) then C_sub_cuda_(a._raw, b._raw) else C_sub_(a._raw, b._raw) end
end

function Tensor.mul_scalar_(a, s)
    if is_cuda(a) then C_mul_scalar_cuda_(a._raw, s) else C_mul_scalar_(a._raw, s) end
end

function Tensor.add_scalar_(a, s)
    if is_cuda(a) then C_add_scalar_cuda_(a._raw, s) else C_add_scalar_(a._raw, s) end
end

-- broadcast add: a is [rows, cols], b is [cols]
-- adds b to every row of a, used for bias in linear layers
function Tensor.add_broadcast(a, b)
    if is_cuda(a) then return wrap_result(C_add_bc_cuda(a._raw, b._raw), a) end
    return wrap_result(C_add_bc(a._raw, b._raw), a)
end

function Tensor.add_broadcast_backward(grad)
    if is_cuda(grad) then return wrap_result(C_add_bc_bwd_cuda(grad._raw), grad) end
    return wrap_result(C_add_bc_bwd(grad._raw), grad)
end

-- matmul ops
function Tensor.matmul(a, b)
    if is_cuda(a) then return wrap_result(C_matmul_cuda(a._raw, b._raw), a) end
    return wrap_result(C_matmul(a._raw, b._raw), a)
end

function Tensor.bmm(a, b)
    if is_cuda(a) then return wrap_result(C_bmm_cuda(a._raw, b._raw), a) end
    return wrap_result(C_bmm(a._raw, b._raw), a)
end

function Tensor.transpose(a)
    local result
    if is_cuda(a) then
        result = wrap_result(C_transpose_cuda(a._raw), a)
    else
        result = wrap_result(C_transpose(a._raw), a)
    end
    if result then result.shape = {a.shape[2], a.shape[1]} end
    return result
end

function Tensor.dot(a, b)
    if is_cuda(a) then return tonumber(C_dot_cuda(a._raw, b._raw)) end
    return tonumber(C_dot(a._raw, b._raw))
end

-- activation ops
function Tensor.relu(a)
    if is_cuda(a) then return wrap_result(C_relu_cuda(a._raw), a) end
    return wrap_result(C_relu(a._raw), a)
end

function Tensor.sigmoid(a)
    if is_cuda(a) then return wrap_result(C_sigmoid_cuda(a._raw), a) end
    return wrap_result(C_sigmoid(a._raw), a)
end

function Tensor.tanh(a)
    if is_cuda(a) then return wrap_result(C_tanh_cuda(a._raw), a) end
    return wrap_result(C_tanh(a._raw), a)
end

function Tensor.gelu(a)
    if is_cuda(a) then return wrap_result(C_gelu_cuda(a._raw), a) end
    return wrap_result(C_gelu(a._raw), a)
end

function Tensor.silu(a)
    if is_cuda(a) then return wrap_result(C_silu_cuda(a._raw), a) end
    return wrap_result(C_silu(a._raw), a)
end

function Tensor.softmax(a)
    if is_cuda(a) then return wrap_result(C_softmax_cuda(a._raw), a) end
    return wrap_result(C_softmax(a._raw), a)
end

function Tensor.gt_scalar(a, s)
    if is_cuda(a) then return wrap_result(C_gt_scalar_cuda(a._raw, s), a) end
    return wrap_result(C_gt_scalar(a._raw, s), a)
end

-- loss ops
function Tensor.mse_loss(pred, target)
    if is_cuda(pred) then return wrap_result(lib.tensor_mse_loss_cuda(pred._raw, target._raw), pred) end
    return wrap_result(lib.tensor_mse_loss(pred._raw, target._raw), pred)
end

function Tensor.mse_loss_backward(pred, target)
    if is_cuda(pred) then return wrap_result(lib.tensor_mse_loss_backward_cuda(pred._raw, target._raw), pred) end
    return wrap_result(lib.tensor_mse_loss_backward(pred._raw, target._raw), pred)
end

function Tensor.mae_loss(pred, target)
    if is_cuda(pred) then return wrap_result(lib.tensor_mae_loss_cuda(pred._raw, target._raw), pred) end
    return wrap_result(lib.tensor_mae_loss(pred._raw, target._raw), pred)
end

function Tensor.mae_loss_backward(pred, target)
    if is_cuda(pred) then return wrap_result(lib.tensor_mae_loss_backward_cuda(pred._raw, target._raw), pred) end
    return wrap_result(lib.tensor_mae_loss_backward(pred._raw, target._raw), pred)
end

function Tensor.cross_entropy_loss(pred, target)
    if is_cuda(pred) then return wrap_result(lib.tensor_cross_entropy_loss_cuda(pred._raw, target._raw), pred) end
    return wrap_result(lib.tensor_cross_entropy_loss(pred._raw, target._raw), pred)
end

function Tensor.cross_entropy_loss_backward(pred, target)
    if is_cuda(pred) then return wrap_result(lib.tensor_cross_entropy_loss_backward_cuda(pred._raw, target._raw), pred) end
    return wrap_result(lib.tensor_cross_entropy_loss_backward(pred._raw, target._raw), pred)
end

-- fused ops (cuda only, fall back to separate ops on cpu)
function Tensor.fused_gelu_bias(x, bias)
    if is_cuda(x) then
        return wrap_result(lib.tensor_fused_gelu_bias_cuda(x._raw, bias._raw), x)
    end
    local added = Tensor.add(x, bias)
    return Tensor.gelu(added)
end

function Tensor.fused_layernorm(x, gamma, beta, eps)
    eps = eps or 1e-5
    if is_cuda(x) then
        return wrap_result(lib.tensor_fused_layernorm_cuda(x._raw, gamma._raw, beta._raw, eps), x)
    end
    return nil  -- cpu uses the Lua layernorm implementation
end

function Tensor.fused_adam(param, grad, m, v, lr, beta1, beta2, eps, bc1, bc2, wd)
    if is_cuda(param) then
        lib.tensor_fused_adam_cuda(param._raw, grad._raw, m._raw, v._raw,
            lr, beta1, beta2, eps, bc1, bc2, wd or 0.0)
        return true
    end
    return false  -- cpu uses the Lua adam implementation
end

-- flash attention
function Tensor.flash_attention(Q, K, V, batch_heads, seq_len, head_dim, causal)
    causal = causal and 1 or 0
    if is_cuda(Q) then
        return wrap_result(lib.tensor_flash_attention_cuda(
            Q._raw, K._raw, V._raw, batch_heads, seq_len, head_dim, causal), Q)
    end
    return nil  -- cpu uses naive attention
end

-- gradient utilities
function Tensor.has_inf_nan(t)
    if is_cuda(t) then
        return lib.tensor_has_inf_nan_cuda(t._raw) ~= 0
    end
    -- cpu version
    for i = 0, t:numel() - 1 do
        local v = t:get(i)
        if v ~= v or v == math.huge or v == -math.huge then return true end
    end
    return false
end

function Tensor.scale_(t, s)
    if is_cuda(t) then
        lib.tensor_scale_cuda(t._raw, s)
    else
        Tensor.mul_scalar_(t, s)
    end
end

-- im2col for conv2d, implemented in C for performance
-- perf fix: moved from pure Lua nested loops to C
function Tensor.im2col(input, batch, channels, height, width, kernel_size, stride, padding)
    return wrap_result(C_im2col(input._raw, batch, channels, height, width,
        kernel_size, stride, padding), input)
end

-- memory pool
function Tensor.pool_init()
    lib.pool_init()
end

function Tensor.pool_clear()
    lib.pool_clear()
end

function Tensor.cuda_memory_allocated()
    return tonumber(lib.cuda_memory_allocated())
end

function Tensor.cuda_memory_cached()
    return tonumber(lib.cuda_memory_cached())
end

-- operator overloads
Tensor.__add = function(a, b) return Tensor.add(a, b) end
Tensor.__sub = function(a, b) return Tensor.sub(a, b) end
Tensor.__mul = function(a, b) return Tensor.mul(a, b) end
Tensor.__div = function(a, b) return Tensor.div(a, b) end
Tensor.__unm = function(a)    return Tensor.neg(a)    end

return Tensor
