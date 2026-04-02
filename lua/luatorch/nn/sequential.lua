local autograd = require('luatorch.autograd')

-- sequential chains layers together in order
-- you add layers and it runs them one after another
-- the output of each layer feeds into the next

local Sequential = {}
Sequential.__index = Sequential

function Sequential.new(...)
    local self = setmetatable({}, Sequential)
    self.layers = {...}
    return self
end

-- add a layer to the end
function Sequential:add(layer)
    table.insert(self.layers, layer)
    return self
end

-- run input through all layers in order
function Sequential:forward(input)
    local out = input
    for _, layer in ipairs(self.layers) do
        out = layer:forward(out)
    end
    return out
end

-- make it callable
Sequential.__call = function(self, input)
    return self:forward(input)
end

-- collect all parameters from all layers
function Sequential:parameters()
    local params = {}
    for _, layer in ipairs(self.layers) do
        if layer.parameters then
            for _, p in ipairs(layer:parameters()) do
                table.insert(params, p)
            end
        end
    end
    return params
end

-- zero gradients on all layers
function Sequential:zero_grad()
    autograd.zero_grad(self:parameters())
end

-- total trainable parameters across all layers
function Sequential:num_params()
    local total = 0
    for _, layer in ipairs(self.layers) do
        if layer.num_params then
            total = total + layer:num_params()
        end
    end
    return total
end

-- print all layers
function Sequential:__tostring()
    local parts = {'Sequential('}
    for i, layer in ipairs(self.layers) do
        table.insert(parts, string.format('  [%d] %s', i, tostring(layer)))
    end
    table.insert(parts, ')')
    return table.concat(parts, '\n')
end

return Sequential
