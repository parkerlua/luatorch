local Tensor   = require('luatorch.tensor')
local autograd = require('luatorch.autograd')

-- batch normalization
-- normalizes across the batch dimension
-- tracks running mean and variance during training
-- uses running stats during inference for consistent behavior
-- learnable gamma (scale) and beta (shift)

-- batchnorm for 1d inputs [batch, features]
local BatchNorm1d = {}
BatchNorm1d.__index = BatchNorm1d

function BatchNorm1d.new(num_features, eps, momentum)
    local self = setmetatable({}, BatchNorm1d)

    self.num_features = num_features
    self.eps          = eps      or 1e-5
    self.momentum     = momentum or 0.1
    self.training     = true

    -- learnable parameters
    self.gamma = Tensor.new({num_features})
    self.gamma:ones()
    autograd.watch(self.gamma)

    self.beta = Tensor.new({num_features})
    self.beta:zeros()
    autograd.watch(self.beta)

    -- running stats for inference
    self.running_mean = Tensor.new({num_features})
    self.running_mean:zeros()
    self.running_var = Tensor.new({num_features})
    self.running_var:ones()

    return self
end

-- forward pass
-- input shape is [batch, features]
function BatchNorm1d:forward(input)
    local batch = input.shape[1]
    local dim   = input.shape[2]
    local out   = Tensor.new(input.shape)

    if self.training then
        -- compute batch statistics
        for f = 0, dim - 1 do
            -- mean
            local sum = 0.0
            for b = 0, batch - 1 do
                sum = sum + input:get(b * dim + f)
            end
            local mean = sum / batch

            -- variance
            local var_sum = 0.0
            for b = 0, batch - 1 do
                local diff = input:get(b * dim + f) - mean
                var_sum = var_sum + diff * diff
            end
            local var = var_sum / batch

            -- normalize and apply scale/shift
            local inv_std = 1.0 / math.sqrt(var + self.eps)
            local gamma   = self.gamma:get(f)
            local beta    = self.beta:get(f)

            for b = 0, batch - 1 do
                local x_norm = (input:get(b * dim + f) - mean) * inv_std
                out:set(b * dim + f, gamma * x_norm + beta)
            end

            -- update running stats
            local old_mean = self.running_mean:get(f)
            local old_var  = self.running_var:get(f)
            self.running_mean:set(f, (1.0 - self.momentum) * old_mean + self.momentum * mean)
            self.running_var:set(f,  (1.0 - self.momentum) * old_var  + self.momentum * var)
        end
    else
        -- use running stats during inference
        for f = 0, dim - 1 do
            local mean    = self.running_mean:get(f)
            local var     = self.running_var:get(f)
            local inv_std = 1.0 / math.sqrt(var + self.eps)
            local gamma   = self.gamma:get(f)
            local beta    = self.beta:get(f)

            for b = 0, batch - 1 do
                local x_norm = (input:get(b * dim + f) - mean) * inv_std
                out:set(b * dim + f, gamma * x_norm + beta)
            end
        end
    end

    -- register backward (similar to layernorm backward)
    autograd.record("batchnorm1d", {input}, out, function(grad)
        if input.requires_grad then
            local dx = Tensor.new(input.shape)

            for f = 0, dim - 1 do
                local sum = 0.0
                for b = 0, batch - 1 do
                    sum = sum + input:get(b * dim + f)
                end
                local mean = sum / batch

                local var_sum = 0.0
                for b = 0, batch - 1 do
                    local diff = input:get(b * dim + f) - mean
                    var_sum = var_sum + diff * diff
                end
                local var = var_sum / batch
                local inv_std = 1.0 / math.sqrt(var + self.eps)

                local x_hat = {}
                local dl_dxhat = {}
                for b = 0, batch - 1 do
                    x_hat[b] = (input:get(b * dim + f) - mean) * inv_std
                    dl_dxhat[b] = grad:get(b * dim + f) * self.gamma:get(f)
                end

                local mean_dl = 0.0
                local mean_dl_xhat = 0.0
                for b = 0, batch - 1 do
                    mean_dl = mean_dl + dl_dxhat[b]
                    mean_dl_xhat = mean_dl_xhat + dl_dxhat[b] * x_hat[b]
                end
                mean_dl = mean_dl / batch
                mean_dl_xhat = mean_dl_xhat / batch

                for b = 0, batch - 1 do
                    dx:set(b * dim + f,
                        inv_std * (dl_dxhat[b] - mean_dl - x_hat[b] * mean_dl_xhat))
                end
            end

            autograd.acc_grad(input, dx)
        end

        if self.gamma.requires_grad then
            local dgamma = Tensor.new({dim})
            dgamma:zeros()
            local dbeta = Tensor.new({dim})
            dbeta:zeros()

            for f = 0, dim - 1 do
                local sum = 0.0
                for b = 0, batch - 1 do
                    sum = sum + input:get(b * dim + f)
                end
                local mean = sum / batch

                local var_sum = 0.0
                for b = 0, batch - 1 do
                    local diff = input:get(b * dim + f) - mean
                    var_sum = var_sum + diff * diff
                end
                local inv_std = 1.0 / math.sqrt(var_sum / batch + self.eps)

                for b = 0, batch - 1 do
                    local x_norm = (input:get(b * dim + f) - mean) * inv_std
                    local g = grad:get(b * dim + f)
                    dgamma:set(f, dgamma:get(f) + g * x_norm)
                    dbeta:set(f, dbeta:get(f) + g)
                end
            end

            autograd.acc_grad(self.gamma, dgamma)
            autograd.acc_grad(self.beta, dbeta)
        end
    end)

    return out
end

BatchNorm1d.__call = function(self, input)
    return self:forward(input)
end

function BatchNorm1d:train() self.training = true end
function BatchNorm1d:eval()  self.training = false end

function BatchNorm1d:parameters()
    return {self.gamma, self.beta}
end

function BatchNorm1d:zero_grad()
    autograd.zero_grad(self:parameters())
end

function BatchNorm1d:num_params()
    return self.num_features * 2
end

function BatchNorm1d:__tostring()
    return string.format('BatchNorm1d(%d)', self.num_features)
end

return {
    BatchNorm1d = BatchNorm1d,
}
