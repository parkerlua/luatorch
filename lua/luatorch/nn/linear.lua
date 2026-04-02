local Tensor   = require('luatorch.tensor')
local autograd = require('luatorch.autograd')

-- a linear layer is the most basic building block of a neural network
-- it does one thing: out = input @ weights + bias
-- thats it. everything in a neural network is stacks of this with activations between

local Linear = {}
Linear.__index = Linear

-- create a new linear layer
-- in_features  = size of each input vector
-- out_features = size of each output vector
-- bias         = whether to add a bias term, true by default
function Linear.new(in_features, out_features, bias)
    local self = setmetatable({}, Linear)

    self.in_features  = in_features
    self.out_features = out_features
    self.use_bias     = bias ~= false  -- default true

    -- initialize weights with random normal distribution
    -- shape is [in_features, out_features]
    -- kaiming initialization scales by sqrt(2/in_features)
    -- this keeps activations from exploding or vanishing as network gets deeper
    self.weight = Tensor.new({in_features, out_features})
    self.weight:randn()
    local scale = math.sqrt(2.0 / in_features)
    self.weight = Tensor.mul_scalar(self.weight, scale)
    autograd.watch(self.weight)

    -- bias starts at zero
    -- shape is [out_features]
    if self.use_bias then
        self.bias = Tensor.new({out_features})
        self.bias:zeros()
        autograd.watch(self.bias)
    end

    return self
end

-- forward pass
-- input shape is [batch_size, in_features]
-- output shape is [batch_size, out_features]
function Linear:forward(input)
    -- main operation: input @ weight
    local out = autograd.matmul(input, self.weight)

    -- add bias if enabled
    -- bias gets added to every row in the batch
    if self.use_bias then
        out = autograd.add(out, self.bias)
    end

    return out
end

-- make the layer callable like a function
-- so you can do layer(input) instead of layer:forward(input)
Linear.__call = function(self, input)
    return self:forward(input)
end

-- return all trainable parameters
-- the optimizer needs this to update weights
function Linear:parameters()
    if self.use_bias then
        return {self.weight, self.bias}
    else
        return {self.weight}
    end
end

-- zero out gradients on all parameters
-- call before each training step
function Linear:zero_grad()
    autograd.zero_grad(self:parameters())
end

-- total number of trainable parameters in this layer
function Linear:num_params()
    local n = self.in_features * self.out_features
    if self.use_bias then
        n = n + self.out_features
    end
    return n
end

-- print layer info
function Linear:__tostring()
    return string.format(
        'Linear(in=%d, out=%d, bias=%s, params=%d)',
        self.in_features,
        self.out_features,
        tostring(self.use_bias),
        self:num_params()
    )
end

return Linear