local Linear     = require('luatorch.nn.linear')
local activation = require('luatorch.nn.activation')
local Sequential = require('luatorch.nn.sequential')
local autograd   = require('luatorch.autograd')

-- simple configurable MLP
-- give it a list of layer sizes and it builds a network
-- relu between every layer, no activation on the output
-- useful for quick baselines and classification heads

local MLP = {}
MLP.__index = MLP

-- sizes = table of layer sizes, e.g. {784, 256, 128, 10}
-- this creates: Linear(784,256) -> ReLU -> Linear(256,128) -> ReLU -> Linear(128,10)
function MLP.new(sizes)
    local self = setmetatable({}, MLP)

    assert(#sizes >= 2, 'luatorch error: MLP needs at least 2 layer sizes')

    self.sizes = sizes
    self.layers = {}

    for i = 1, #sizes - 1 do
        table.insert(self.layers, Linear.new(sizes[i], sizes[i + 1]))
        -- add relu between layers, but not after the last one
        if i < #sizes - 1 then
            table.insert(self.layers, activation.ReLU.new())
        end
    end

    self.model = Sequential.new(table.unpack(self.layers))

    return self
end

function MLP:forward(input)
    return self.model:forward(input)
end

MLP.__call = function(self, input)
    return self:forward(input)
end

function MLP:parameters()
    return self.model:parameters()
end

function MLP:zero_grad()
    self.model:zero_grad()
end

function MLP:num_params()
    return self.model:num_params()
end

function MLP:__tostring()
    return string.format('MLP(%s)\n%s',
        table.concat(self.sizes, '->'),
        tostring(self.model))
end

return MLP
