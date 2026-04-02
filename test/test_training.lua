local Tensor     = require('luatorch.tensor')
local autograd   = require('luatorch.autograd')
local Linear     = require('luatorch.nn.linear')
local activation = require('luatorch.nn.activation')
local loss_fn    = require('luatorch.nn.loss')
local Sequential = require('luatorch.nn.sequential')
local Adam       = require('luatorch.optim.adam')
local checkpoint = require('luatorch.io.checkpoint')

-- XOR training converges to loss below 0.05 in 2000 steps
do
    local inputs = Tensor.new({4, 2})
    inputs:set(0, 0); inputs:set(1, 0)
    inputs:set(2, 0); inputs:set(3, 1)
    inputs:set(4, 1); inputs:set(5, 0)
    inputs:set(6, 1); inputs:set(7, 1)

    local targets = Tensor.new({4, 1})
    targets:set(0, 0); targets:set(1, 1)
    targets:set(2, 1); targets:set(3, 0)

    local model = Sequential.new(
        Linear.new(2, 16),
        activation.ReLU.new(),
        Linear.new(16, 1),
        activation.Sigmoid.new()
    )

    local params    = model:parameters()
    local optimizer = Adam.new(params, 0.01)
    local criterion = loss_fn.MSELoss.new()

    local final_loss = 999
    for step = 1, 2000 do
        autograd.zero_graph()
        autograd.zero_grad(params)
        autograd.watch(inputs)

        local pred = model(inputs)
        local loss = criterion(pred, targets)
        autograd.backward(loss)
        optimizer:step()

        final_loss = loss:get(0)
    end

    assert(final_loss < 0.05,
        string.format('xor should converge, final loss: %.4f', final_loss))
end

-- Checkpoint save then load produces same output
do
    autograd.zero_graph()
    local model = Sequential.new(
        Linear.new(4, 3),
        activation.ReLU.new(),
        Linear.new(3, 2)
    )

    local input = Tensor.new({1, 4})
    input:set(0, 1.0); input:set(1, 2.0); input:set(2, 3.0); input:set(3, 4.0)

    autograd.enabled = false
    local out1 = model(input)
    autograd.enabled = true

    -- save
    local path = '/tmp/luatorch_test_ckpt.bin'
    checkpoint.save(model, path)

    -- create new model with same architecture and load
    local model2 = Sequential.new(
        Linear.new(4, 3),
        activation.ReLU.new(),
        Linear.new(3, 2)
    )
    checkpoint.load(model2, path)

    autograd.enabled = false
    local out2 = model2(input)
    autograd.enabled = true

    for i = 0, out1:numel() - 1 do
        local diff = math.abs(out1:get(i) - out2:get(i))
        assert(diff < 1e-5,
            string.format('checkpoint output mismatch at %d: %f vs %f',
                i, out1:get(i), out2:get(i)))
    end

    os.remove(path)
end

-- train/eval mode toggle
do
    local Dropout = require('luatorch.nn.dropout')
    local drop = Dropout.new(0.5)

    assert(drop.training == true, 'should start in train mode')
    drop:eval()
    assert(drop.training == false, 'should be in eval mode')
    drop:train()
    assert(drop.training == true, 'should be back in train mode')
end
