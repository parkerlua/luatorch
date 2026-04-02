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
local function im2col(input, batch, channels, height, width,
                      kernel_size, stride, padding)
    local out_h = math.floor((height + 2 * padding - kernel_size) / stride) + 1
    local out_w = math.floor((width  + 2 * padding - kernel_size) / stride) + 1
    local col_len = channels * kernel_size * kernel_size

    local cols = Tensor.new({batch * out_h * out_w, col_len})

    for b = 0, batch - 1 do
        for oh = 0, out_h - 1 do
            for ow = 0, out_w - 1 do
                local row = b * out_h * out_w + oh * out_w + ow
                local col = 0
                for c = 0, channels - 1 do
                    for kh = 0, kernel_size - 1 do
                        for kw = 0, kernel_size - 1 do
                            local ih = oh * stride + kh - padding
                            local iw = ow * stride + kw - padding
                            local val = 0.0
                            if ih >= 0 and ih < height and iw >= 0 and iw < width then
                                local idx = ((b * channels + c) * height + ih) * width + iw
                                val = input:get(idx)
                            end
                            cols:set(row * col_len + col, val)
                            col = col + 1
                        end
                    end
                end
            end
        end
    end

    return cols, out_h, out_w
end

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

    -- im2col: turn patches into a matrix
    local cols = im2col(input, batch, in_c, height, width, ksize, s, pad)

    -- matmul: weight @ cols^T -> [out_c, batch * out_h * out_w]
    -- but our matmul expects [M, K] x [K, N]
    -- cols is [batch*out_h*out_w, in_c*kh*kw]
    -- weight is [out_c, in_c*kh*kw]
    -- so we do: cols @ weight^T -> [batch*out_h*out_w, out_c]
    local wt  = Tensor.transpose(self.weight)
    local out = autograd.matmul(cols, wt)

    -- add bias to each output channel
    local total_spatial = batch * out_h * out_w
    for i = 0, total_spatial - 1 do
        for c = 0, out_c - 1 do
            local idx = i * out_c + c
            local val = out:get(idx) + self.bias:get(c)
            out:set(idx, val)
        end
    end

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
