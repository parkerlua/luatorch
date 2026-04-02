local Tensor = require('luatorch.tensor')

-- adam is the most common optimizer used in modern AI
-- it stands for adaptive moment estimation
-- it tracks two things per parameter:
-- m = momentum, which direction have we been moving
-- v = velocity, how fast have we been moving
-- this makes it much smarter than basic gradient descent
-- which just blindly moves in the gradient direction every step

local Adam = {}
Adam.__index = Adam

-- params       = list of tensors to optimize, your model weights
-- lr           = learning rate, how big each step is, 0.001 is a good default
-- beta1        = momentum decay, how much to remember past direction, default 0.9
-- beta2        = velocity decay, how much to remember past speed, default 0.999
-- epsilon      = tiny number to prevent division by zero, default 1e-8
function Adam.new(params, lr, beta1, beta2, epsilon)
    local self = setmetatable({}, Adam)

    self.params  = params
    self.lr      = lr      or 0.001
    self.beta1   = beta1   or 0.9
    self.beta2   = beta2   or 0.999
    self.epsilon = epsilon or 1e-8
    self.t       = 0

    -- initialize momentum and velocity for each parameter
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

-- perform one optimization step
function Adam:step()
    self.t = self.t + 1

    local bc1 = 1.0 - self.beta1 ^ self.t
    local bc2 = 1.0 - self.beta2 ^ self.t

    for i, param in ipairs(self.params) do
        if not param.grad then goto continue end
        local grad = param.grad

        -- use fused cuda kernel when on gpu
        -- does the entire adam update in one kernel launch instead of 7
        if param.device == 'cuda' and Tensor.fused_adam(
            param, grad, self.m[i], self.v[i],
            self.lr, self.beta1, self.beta2, self.epsilon,
            bc1, bc2, 0.0
        ) then
            goto continue
        end

        -- cpu path: separate tensor operations
        Tensor.mul_scalar_(self.m[i], self.beta1)
        local grad_scaled = Tensor.mul_scalar(grad, 1.0 - self.beta1)
        Tensor.add_(self.m[i], grad_scaled)

        Tensor.mul_scalar_(self.v[i], self.beta2)
        local grad_sq = Tensor.pow_scalar(grad, 2.0)
        local grad_sq_scaled = Tensor.mul_scalar(grad_sq, 1.0 - self.beta2)
        Tensor.add_(self.v[i], grad_sq_scaled)

        local m_hat = Tensor.div_scalar(self.m[i], bc1)
        local v_hat = Tensor.div_scalar(self.v[i], bc2)

        local v_sqrt   = Tensor.sqrt(v_hat)
        local v_stable = Tensor.add_scalar(v_sqrt, self.epsilon)
        local update   = Tensor.div(m_hat, v_stable)
        update         = Tensor.mul_scalar(update, self.lr)

        Tensor.sub_(param, update)

        ::continue::
    end
end

function Adam:reset()
    self.t = 0
    for i, _ in ipairs(self.params) do
        self.m[i]:zeros()
        self.v[i]:zeros()
    end
end

function Adam:set_lr(lr)
    self.lr = lr
end

function Adam:__tostring()
    return string.format('Adam(lr=%.6f, beta1=%.3f, beta2=%.3f)',
        self.lr, self.beta1, self.beta2)
end

return Adam
