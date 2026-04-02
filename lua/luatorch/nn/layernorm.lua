local Tensor   = require('luatorch.tensor')
local autograd = require('luatorch.autograd')

-- layer normalization
-- normalizes across the last dimension of the input
-- then applies learnable scale (gamma) and shift (beta)
-- used in every transformer block
-- unlike batch norm, this works the same in training and inference

local LayerNorm = {}
LayerNorm.__index = LayerNorm

-- normalized_shape is the size of the last dimension
-- eps prevents division by zero in the variance calculation
function LayerNorm.new(normalized_shape, eps)
    local self = setmetatable({}, LayerNorm)

    self.normalized_shape = normalized_shape
    self.eps = eps or 1e-5

    -- learnable scale, initialized to 1
    self.gamma = Tensor.new({normalized_shape})
    self.gamma:ones()
    autograd.watch(self.gamma)

    -- learnable shift, initialized to 0
    self.beta = Tensor.new({normalized_shape})
    self.beta:zeros()
    autograd.watch(self.beta)

    return self
end

-- forward pass
-- input shape is [batch, features] or [batch, seq, features]
-- normalizes over the last dimension
function LayerNorm:forward(input)
    local shape = input.shape
    local ndim  = input.ndim
    local dim   = shape[ndim]

    local rows = input:numel() / dim
    local flat = input
    local out = Tensor.new(shape)

    -- perf fix: cache per-row mean, inv_std, and x_hat during forward
    -- old code recomputed them 3 separate times in backward, wasting ~3x compute
    local cached_inv_std = {}
    local cached_x_hat   = {}

    for r = 0, rows - 1 do
        local row_sum = 0.0
        for c = 0, dim - 1 do
            row_sum = row_sum + flat:get(r * dim + c)
        end
        local row_mean = row_sum / dim

        local var_sum = 0.0
        for c = 0, dim - 1 do
            local diff = flat:get(r * dim + c) - row_mean
            var_sum = var_sum + diff * diff
        end
        local inv_std = 1.0 / math.sqrt(var_sum / dim + self.eps)

        cached_inv_std[r] = inv_std

        local x_hat_row = {}
        for c = 0, dim - 1 do
            local x_norm = (flat:get(r * dim + c) - row_mean) * inv_std
            x_hat_row[c] = x_norm
            out:set(r * dim + c, self.gamma:get(c) * x_norm + self.beta:get(c))
        end
        cached_x_hat[r] = x_hat_row
    end

    autograd.record("layernorm", {input}, out, function(grad)
        if input.requires_grad then
            local dx = Tensor.new(shape)

            for r = 0, rows - 1 do
                -- use cached values instead of recomputing
                local inv_std  = cached_inv_std[r]
                local x_hat   = cached_x_hat[r]

                local dl_dxhat = {}
                for c = 0, dim - 1 do
                    dl_dxhat[c] = grad:get(r * dim + c) * self.gamma:get(c)
                end

                local mean_dl = 0.0
                local mean_dl_xhat = 0.0
                for c = 0, dim - 1 do
                    mean_dl = mean_dl + dl_dxhat[c]
                    mean_dl_xhat = mean_dl_xhat + dl_dxhat[c] * x_hat[c]
                end
                mean_dl = mean_dl / dim
                mean_dl_xhat = mean_dl_xhat / dim

                for c = 0, dim - 1 do
                    dx:set(r * dim + c,
                        inv_std * (dl_dxhat[c] - mean_dl - x_hat[c] * mean_dl_xhat))
                end
            end

            autograd.acc_grad(input, dx)
        end

        if self.gamma.requires_grad then
            local dgamma = Tensor.new({dim})
            dgamma:zeros()
            local dbeta = Tensor.new({dim})
            dbeta:zeros()

            for r = 0, rows - 1 do
                local x_hat = cached_x_hat[r]
                for c = 0, dim - 1 do
                    local g = grad:get(r * dim + c)
                    dgamma:set(c, dgamma:get(c) + g * x_hat[c])
                    dbeta:set(c, dbeta:get(c) + g)
                end
            end

            autograd.acc_grad(self.gamma, dgamma)
            autograd.acc_grad(self.beta, dbeta)
        end
    end)

    return out
end

LayerNorm.__call = function(self, input)
    return self:forward(input)
end

function LayerNorm:parameters()
    return {self.gamma, self.beta}
end

function LayerNorm:zero_grad()
    autograd.zero_grad(self:parameters())
end

function LayerNorm:num_params()
    return self.normalized_shape * 2
end

function LayerNorm:__tostring()
    return string.format('LayerNorm(%d)', self.normalized_shape)
end

return LayerNorm
