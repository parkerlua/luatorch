local Tensor   = require('luatorch.tensor')
local autograd = require('luatorch.autograd')
local Adam     = require('luatorch.optim.adam')
local AdamW    = require('luatorch.optim.adamw')
local SGD      = require('luatorch.optim.sgd')
local sched    = require('luatorch.optim.scheduler')

local function assert_near(a, b, tol, msg)
    tol = tol or 1e-3
    if math.abs(a - b) > tol then
        error(string.format('%s: expected ~%f got %f', msg or 'assert_near', b, a))
    end
end

-- Adam step reduces a simple quadratic loss
-- minimize f(x) = x^2, gradient is 2x
do
    local x = Tensor.new({1}); x:set(0, 5.0)
    autograd.watch(x)

    local optimizer = Adam.new({x}, 0.1)

    for _ = 1, 100 do
        autograd.zero_graph()
        autograd.zero_grad({x})

        -- forward: loss = x^2
        local loss = Tensor.mul(x, x)
        autograd.backward(loss)

        optimizer:step()
    end

    assert(math.abs(x:get(0)) < 0.5,
        string.format('adam should minimize x^2, got x=%f', x:get(0)))
end

-- AdamW applies weight decay
do
    local x = Tensor.new({1}); x:set(0, 10.0)
    autograd.watch(x)

    local optimizer = AdamW.new({x}, 0.1, 0.9, 0.999, 1e-8, 0.1)

    local initial = x:get(0)
    -- do one step with zero gradient
    autograd.zero_graph()
    autograd.zero_grad({x})
    x.grad = Tensor.new({1}); x.grad:zeros()
    optimizer:step()

    -- weight decay should have reduced x even with zero gradient
    assert(x:get(0) < initial,
        string.format('adamw weight decay: expected < %f got %f', initial, x:get(0)))
end

-- Schedulers change lr correctly
do
    local x = Tensor.new({1}); x:set(0, 1.0)
    autograd.watch(x)
    local opt = Adam.new({x}, 0.1)

    -- cosine annealing
    local cosine = sched.CosineAnnealing.new(opt, 100, 0.001)
    local lr1 = cosine:step()
    local lr2 = cosine:step()
    assert(lr1 > lr2, 'cosine lr should decrease')

    -- step lr
    opt.lr = 0.1
    local step_sched = sched.StepLR.new(opt, 2, 0.5)
    step_sched:step()  -- epoch 1, no decay
    assert_near(opt.lr, 0.1, 1e-6, 'step lr epoch 1')
    step_sched:step()  -- epoch 2, decay
    assert_near(opt.lr, 0.05, 1e-6, 'step lr epoch 2')

    -- warmup
    opt.lr = 0.1
    local warmup = sched.WarmupScheduler.new(opt, 10)
    local wlr = warmup:step()
    assert(wlr < 0.1, 'warmup lr should start below target')
    assert(wlr > 0, 'warmup lr should be positive')
end
