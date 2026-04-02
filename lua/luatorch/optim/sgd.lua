local Tensor = require('luatorch.tensor')

-- stochastic gradient descent with momentum and optional nesterov
-- the simplest optimizer that actually works well
-- momentum helps it power through noisy gradients
-- nesterov looks ahead before computing the gradient direction

local SGD = {}
SGD.__index = SGD

-- params   = list of tensors to optimize
-- lr       = learning rate, default 0.01
-- momentum = momentum factor, default 0.9
-- nesterov = use nesterov momentum, default false
function SGD.new(params, lr, momentum, nesterov)
    local self = setmetatable({}, SGD)

    self.params   = params
    self.lr       = lr       or 0.01
    self.momentum = momentum or 0.9
    self.nesterov = nesterov or false

    -- initialize velocity buffers
    self.v = {}
    for i, param in ipairs(self.params) do
        self.v[i] = Tensor.new(param.shape)
        self.v[i]:zeros()
    end

    return self
end

function SGD:step()
    for i, param in ipairs(self.params) do
        if not param.grad then goto continue end
        local grad = param.grad

        -- update velocity: v = momentum * v + grad
        Tensor.mul_scalar_(self.v[i], self.momentum)
        Tensor.add_(self.v[i], grad)

        if self.nesterov then
            -- nesterov: param = param - lr * (momentum * v + grad)
            local lookahead = Tensor.mul_scalar(self.v[i], self.momentum)
            local update    = Tensor.add(lookahead, grad)
            update          = Tensor.mul_scalar(update, self.lr)
            Tensor.sub_(param, update)
        else
            -- standard: param = param - lr * v
            local update = Tensor.mul_scalar(self.v[i], self.lr)
            Tensor.sub_(param, update)
        end

        ::continue::
    end
end

function SGD:reset()
    for i, _ in ipairs(self.params) do
        self.v[i]:zeros()
    end
end

function SGD:set_lr(lr)
    self.lr = lr
end

function SGD:__tostring()
    return string.format('SGD(lr=%.6f, momentum=%.3f, nesterov=%s)',
        self.lr, self.momentum, tostring(self.nesterov))
end

return SGD
