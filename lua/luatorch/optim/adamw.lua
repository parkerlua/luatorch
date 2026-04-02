local Tensor = require('luatorch.tensor')

-- adamw is adam with decoupled weight decay
-- the standard optimizer for training transformers
-- weight decay is applied directly to weights, not through the gradient
-- this gives better generalization than L2 regularization in adam
-- bias and layernorm parameters should not be decayed

local AdamW = {}
AdamW.__index = AdamW

-- params        = list of tensors to optimize
-- lr            = learning rate, default 0.001
-- beta1         = momentum decay, default 0.9
-- beta2         = velocity decay, default 0.999
-- epsilon       = stability constant, default 1e-8
-- weight_decay  = decay factor, default 0.01
-- no_decay_keys = list of parameter tensors to skip weight decay on
function AdamW.new(params, lr, beta1, beta2, epsilon, weight_decay, no_decay_keys)
    local self = setmetatable({}, AdamW)

    self.params       = params
    self.lr           = lr           or 0.001
    self.beta1        = beta1        or 0.9
    self.beta2        = beta2        or 0.999
    self.epsilon      = epsilon      or 1e-8
    self.weight_decay = weight_decay or 0.01
    self.t            = 0

    -- build a set of parameters that should skip weight decay
    self.no_decay = {}
    if no_decay_keys then
        for _, p in ipairs(no_decay_keys) do
            self.no_decay[p] = true
        end
    end

    -- initialize momentum and velocity
    self.m = {}
    self.v = {}
    for i, param in ipairs(self.params) do
        self.m[i] = Tensor.new(param.shape)
        self.m[i]:zeros()
        self.v[i] = Tensor.new(param.shape)
        self.v[i]:zeros()
    end

    return self
end

function AdamW:step()
    self.t = self.t + 1

    local bc1 = 1.0 - self.beta1 ^ self.t
    local bc2 = 1.0 - self.beta2 ^ self.t

    for i, param in ipairs(self.params) do
        if not param.grad then goto continue end
        local grad = param.grad

        -- apply weight decay before the adam update
        -- param = param * (1 - lr * weight_decay)
        -- skip for parameters marked as no_decay (bias, layernorm)
        if self.weight_decay > 0 and not self.no_decay[param] then
            local decay = 1.0 - self.lr * self.weight_decay
            Tensor.mul_scalar_(param, decay)
        end

        -- update momentum: m = beta1 * m + (1 - beta1) * grad
        Tensor.mul_scalar_(self.m[i], self.beta1)
        local grad_scaled = Tensor.mul_scalar(grad, 1.0 - self.beta1)
        Tensor.add_(self.m[i], grad_scaled)

        -- update velocity: v = beta2 * v + (1 - beta2) * grad^2
        Tensor.mul_scalar_(self.v[i], self.beta2)
        local grad_sq = Tensor.pow_scalar(grad, 2.0)
        local grad_sq_scaled = Tensor.mul_scalar(grad_sq, 1.0 - self.beta2)
        Tensor.add_(self.v[i], grad_sq_scaled)

        -- bias corrected estimates
        local m_hat = Tensor.div_scalar(self.m[i], bc1)
        local v_hat = Tensor.div_scalar(self.v[i], bc2)

        -- compute update: lr * m_hat / (sqrt(v_hat) + eps)
        local v_sqrt   = Tensor.sqrt(v_hat)
        local v_stable = Tensor.add_scalar(v_sqrt, self.epsilon)
        local update   = Tensor.div(m_hat, v_stable)
        update         = Tensor.mul_scalar(update, self.lr)

        Tensor.sub_(param, update)

        ::continue::
    end
end

function AdamW:reset()
    self.t = 0
    for i, _ in ipairs(self.params) do
        self.m[i]:zeros()
        self.v[i]:zeros()
    end
end

function AdamW:set_lr(lr)
    self.lr = lr
end

function AdamW:__tostring()
    return string.format('AdamW(lr=%.6f, wd=%.4f, beta1=%.3f, beta2=%.3f)',
        self.lr, self.weight_decay, self.beta1, self.beta2)
end

return AdamW
