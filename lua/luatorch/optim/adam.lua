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

-- create a new adam optimizer
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
    self.t       = 0       -- step counter, used for bias correction

    -- initialize momentum and velocity for each parameter
    -- both start at zero
    self.m = {}  -- first moment, momentum
    self.v = {}  -- second moment, velocity

    for i, param in ipairs(self.params) do
        self.m[i] = Tensor.new(param.shape)
        self.m[i]:zeros()
        self.v[i] = Tensor.new(param.shape)
        self.v[i]:zeros()
    end

    return self
end

-- perform one optimization step
-- call this after loss:backward()
function Adam:step()
    -- increment step counter
    self.t = self.t + 1

    -- bias correction terms
    -- early in training m and v are biased toward zero
    -- these terms correct for that
    local bc1 = 1.0 - self.beta1 ^ self.t
    local bc2 = 1.0 - self.beta2 ^ self.t

    for i, param in ipairs(self.params) do
        -- skip if no gradient, parameter wasnt used in forward pass
        if not param.grad then goto continue end

        local grad = param.grad

        -- update momentum
        -- m = beta1 * m + (1 - beta1) * grad
        -- blends old direction with new gradient
        Tensor.mul_scalar_(self.m[i], self.beta1)
        local grad_scaled = Tensor.mul_scalar(grad, 1.0 - self.beta1)
        Tensor.add_(self.m[i], grad_scaled)

        -- update velocity
        -- v = beta2 * v + (1 - beta2) * grad^2
        -- tracks how noisy each gradient has been
        Tensor.mul_scalar_(self.v[i], self.beta2)
        local grad_sq     = Tensor.pow_scalar(grad, 2.0)
        local grad_sq_scaled = Tensor.mul_scalar(grad_sq, 1.0 - self.beta2)
        Tensor.add_(self.v[i], grad_sq_scaled)

        -- bias corrected estimates
        local m_hat = Tensor.div_scalar(self.m[i], bc1)
        local v_hat = Tensor.div_scalar(self.v[i], bc2)

        -- compute the update
        -- update = lr * m_hat / (sqrt(v_hat) + epsilon)
        -- parameters that have been moving consistently get bigger updates
        -- parameters that have been noisy get smaller updates
        local v_sqrt   = Tensor.sqrt(v_hat)
        local v_stable = Tensor.add_scalar(v_sqrt, self.epsilon)
        local update   = Tensor.div(m_hat, v_stable)
        update         = Tensor.mul_scalar(update, self.lr)

        -- subtract update from parameter
        -- we subtract because we want to minimize the loss
        Tensor.sub_(param, update)

        ::continue::
    end
end

-- reset optimizer state
-- call this if you want to start training fresh
function Adam:reset()
    self.t = 0
    for i, param in ipairs(self.params) do
        self.m[i]:zeros()
        self.v[i]:zeros()
    end
end

-- cha