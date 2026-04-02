local Tensor   = require('luatorch.tensor')
local autograd = require('luatorch.autograd')

-- 2d convolution layer
-- slides a kernel over the input image and computes dot products
-- the fundamental building block for computer vision
-- uses im2col: unrolls image patches into columns for matmul

local Conv2d = {}
Conv2d.__index = Conv2d

-- in_channels  = input channels (e.g. 3 for RGB)
-- out_channels = number of filters (e.g. 32)
-- kernel_size  = filter size (e.g. 3 for 3x3)
-- stride       = step size, default 1
-- padding      = zero padding, default 0
function Conv2d.new(in_channels, out_channels, kernel_size, stride, padding)
    local self = setmetatable({}, Conv2d)

    self.in_channels  = in_channels
    self.out_channels = out_channels
    self.kernel_size  = kernel_size
    self.stride       = stride  or 1
    self.padding      = padding or 0

    -- weight shape is [out_channels, in_channels * kernel_size * kernel_size]
    -- stored flat for matmul with im2col columns
    local fan_in = in_channels * kernel_size * kernel_size
    self.weight = Tensor.new({out_channels, fan_in})
    self.weight:randn()
    local scale = math.sqrt(2.0 / fan_in)
    self.weight = Tensor.mul_scalar(self.weight, scale)
    autograd.watch(self.weight)

    -- bias shape is [out_channels]
    self.bias = Tensor.new({out_channels})
    self.bias:zeros()
    autograd.watch(self.bias)

    return self
end

-- im2col: unroll image patches into columns
-- input shape is [batch, in_channels, height, width] stored flat
-- output is [batch * out_h * out_w, in_channels * kh * kw]
-- perf fix: im2col moved to C (csrc/ops/conv2d.c) for ~100x speedup
-- the old Lua version used 6 nested loops with element-by-element get/set

-- forward pass
-- input is [batch, in_channels, height, width] stored as flat tensor
-- shapes must be provided since we work with flat tensors
function Conv2d:forward(input, batch, height, width)
    local ksize    = self.kernel_size
    local s        = self.stride
    local pad      = self.padding
    local in_c     = self.in_channels
    local out_c    = self.out_channels

    local out_h = math.floor((height + 2 * pad - ksize) / s) + 1
    local out_w = math.floor((width  + 2 * pad - ksize) / s) + 1

    -- im2col: turn patches into a matrix (C implementation)
    local cols = Tensor.im2col(input, batch, in_c, height, width, ksize, s, pad)

    -- matmul: weight @ cols^T -> [out_c, batch * out_h * out_w]
    -- but our matmul expects [M, K] x [K, N]
    -- cols is [batch*out_h*out_w, in_c*kh*kw]
    -- weight is [out_c, in_c*kh*kw]
    -- so we do: cols @ weight^T -> [batch*out_h*out_w, out_c]
    local wt  = Tensor.transpose(self.weight)
    local out = autograd.matmul(cols, wt)

    -- fix: add bias through broadcast add with autograd so bias gradient flows
    -- out is [batch*out_h*out_w, out_c], bias is [out_c]
    local biased = Tensor.add_broadcast(out, self.bias)

    autograd.record("conv2d_bias", {out, self.bias}, biased, function(grad)
        if out.requires_grad then
            autograd.acc_grad(out, grad)
        end
        if self.bias.requires_grad then
            autograd.acc_grad(self.bias, Tensor.add_broadcast_backward(grad))
        end
    end)

    out = biased

    -- store metadata for backward and for downstream layers
    out._conv_batch = batch
    out._conv_out_h = out_h
    out._conv_out_w = out_w
    out._conv_out_c = out_c

    return out, out_h, out_w
end

Conv2d.__call = function(self, input, batch, height, width)
    return self:forward(input, batch, height, width)
end

function Conv2d:parameters()
    return {self.weight, self.bias}
end

function Conv2d:zero_grad()
    autograd.zero_grad(self:parameters())
end

function Conv2d:num_params()
    return self.out_channels * self.in_channels * self.kernel_size * self.kernel_size
           + self.out_channels
end

function Conv2d:__tostring()
    return string.format('Conv2d(in=%d, out=%d, kernel=%d, stride=%d, pad=%d, params=%d)',
        self.in_channels, self.out_channels, self.kernel_size,
        self.stride, self.padding, self:num_params())
end

return Conv2d
