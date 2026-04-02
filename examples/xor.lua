-- xor neural network
-- the classic first test for any AI framework
-- xor cant be solved by a single linear layer
-- you need at least one hidden layer with a nonlinearity

local Tensor     = require('luatorch.tensor')
local autograd   = require('luatorch.autograd')
local Linear     = require('luatorch.nn.linear')
local activation = require('luatorch.nn.activation')
local loss_fn    = require('luatorch.nn.loss')
local Sequential = require('luatorch.nn.sequential')
local Adam       = require('luatorch.optim.adam')

-- xor truth table
-- inputs:  [0,0] [0,1] [1,0] [1,1]
-- outputs: [0]   [1]   [1]   [0]
local inputs = Tensor.new({4, 2})
inputs:set(0, 0.0) inputs:set(1, 0.0)  -- [0, 0]
inputs:set(2, 0.0) inputs:set(3, 1.0)  -- [0, 1]
inputs:set(4, 1.0) inputs:set(5, 0.0)  -- [1, 0]
inputs:set(6, 1.0) inputs:set(7, 1.0)  -- [1, 1]

local targets = Tensor.new({4, 1})
targets:set(0, 0.0)  -- 0 xor 0 = 0
targets:set(1, 1.0)  -- 0 xor 1 = 1
targets:set(2, 1.0)  -- 1 xor 0 = 1
targets:set(3, 0.0)  -- 1 xor 1 = 0

-- build the network
-- 2 inputs -> 8 hidden neurons -> 1 output
-- relu in the middle to add nonlinearity
local model = Sequential.new(
    Linear.new(2, 8),
    activation.ReLU.new(),
    Linear.new(8, 1),
    activation.Sigmoid.new()
)

print(tostring(model))
print(string.format('total parameters: %d', model:num_params()))

-- setup optimizer and loss
local params    = model:parameters()
local optimizer = Adam.new(params, 0.01)
local criterion = loss_fn.MSELoss.new()

-- training loop
local epochs = 1000
for epoch = 1, epochs do
    -- clear old gradients and computation graph
    autograd.zero_graph()
    model:zero_grad()

    -- forward pass
    autograd.watch(inputs)
    local pred = model(inputs)
    local loss = criterion(pred, targets)

    -- backward pass
    autograd.backward(loss)

    -- update weights
    optimizer:step()

    -- print progress every 100 epochs
    if epoch % 100 == 0 or epoch == 1 then
        local loss_val = loss:get(0)
        print(string.format('epoch %4d  loss: %.6f', epoch, loss_val))
    end
end

-- test the trained network
print('\nresults:')
autograd.enabled = false  -- no need to track gradients for inference
for i = 0, 3 do
    local x1 = inputs:get(i * 2)
    local x2 = inputs:get(i * 2 + 1)
    local target = targets:get(i)

    -- make a single input tensor
    local single = Tensor.new({1, 2})
    single:set(0, x1)
    single:set(1, x2)

    local pred = model(single)
    local pred_val = pred:get(0)
    print(string.format('  %.0f xor %.0f = %.4f (target: %.0f)', x1, x2, pred_val, target))
end
autograd.enabled = true
