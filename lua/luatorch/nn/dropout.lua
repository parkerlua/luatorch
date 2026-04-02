local Tensor   = require('luatorch.tensor')
local autograd = require('luatorch.autograd')

-- dropout randomly zeros out elements during training
-- this forces the network to not rely on any single neuron
-- at test time dropout is disabled and outputs are unchanged
-- during training, surviving elements are scaled by 1/(1-p)
-- so the expected value stays the same

local Dropout = {}
Dropout.__index = Dropout

-- p is the probability of zeroing an element
-- default 0.5 which means half the neurons are dropped each forward pass
function Dropout.new(p)
    local self = setmetatable({}, Dropout)
    self.p        = p or 0.5
    self.training = true
    return self
end

function Dropout:forward(input)
    -- during inference just pass through unchanged
    if not self.training then
        return input
    end

    -- generate random mask
    -- each element has probability (1-p) of surviving
    local mask = Tensor.new(input.shape)
    mask:rand()
    mask = Tensor.gt_scalar(mask, self.p)

    -- scale by 1/(1-p) so expected values dont change
    local scale = 1.0 / (1.0 - self.p)
    local scaled_mask = Tensor.mul_scalar(mask, scale)

    local out = Tensor.mul(input, scaled_mask)

    -- register backward pass
    autograd.record("dropout", {input}, out, function(grad)
        if input.requires_grad then
            -- gradient flows through the same mask
            -- dropped elements get zero gradient
            autograd.acc_grad(input, Tensor.mul(grad, scaled_mask))
        end
    end)

    return out
end

-- switch between training and eval mode
function Dropout:train()
    self.training = true
end

function Dropout:eval()
    self.training = false
end

Dropout.__call = function(self, input)
    return self:forward(input)
end

function Dropout:parameters() return {} end
function Dropout:zero_grad() end

function Dropout:__tostring()
    return string.format('Dropout(p=%.2f)', self.p)
end

return Dropout
