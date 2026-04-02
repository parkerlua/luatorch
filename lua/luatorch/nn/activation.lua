local Tensor   = require('luatorch.tensor')
local autograd = require('luatorch.autograd')

-- activation modules that can be used as layers in a Sequential model
-- each one wraps the C activation function and registers autograd

local ReLU = {}
ReLU.__index = ReLU

function ReLU.new()
    return setmetatable({}, ReLU)
end

function ReLU:forward(input)
    return autograd.relu(input)
end

ReLU.__call = function(self, input)
    return self:forward(input)
end

function ReLU:parameters() return {} end
function ReLU:zero_grad() end
function ReLU:__tostring() return 'ReLU()' end

-- sigmoid activation
local Sigmoid = {}
Sigmoid.__index = Sigmoid

function Sigmoid.new()
    return setmetatable({}, Sigmoid)
end

function Sigmoid:forward(input)
    local out = Tensor.sigmoid(input)

    autograd.record("sigmoid", {input}, out, function(grad)
        if input.requires_grad then
            -- d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
            local ones = Tensor.new(out.shape)
            ones:ones()
            local one_minus = Tensor.sub(ones, out)
            local dsig = Tensor.mul(out, one_minus)
            autograd.acc_grad(input, Tensor.mul(grad, dsig))
        end
    end)

    return out
end

Sigmoid.__call = function(self, input)
    return self:forward(input)
end

function Sigmoid:parameters() return {} end
function Sigmoid:zero_grad() end
function Sigmoid:__tostring() return 'Sigmoid()' end

-- tanh activation
local Tanh = {}
Tanh.__index = Tanh

function Tanh.new()
    return setmetatable({}, Tanh)
end

function Tanh:forward(input)
    local out = Tensor.tanh(input)

    autograd.record("tanh", {input}, out, function(grad)
        if input.requires_grad then
            -- d(tanh)/dx = 1 - tanh(x)^2
            local sq = Tensor.mul(out, out)
            local ones = Tensor.new(out.shape)
            ones:ones()
            local dtanh = Tensor.sub(ones, sq)
            autograd.acc_grad(input, Tensor.mul(grad, dtanh))
        end
    end)

    return out
end

Tanh.__call = function(self, input)
    return self:forward(input)
end

function Tanh:parameters() return {} end
function Tanh:zero_grad() end
function Tanh:__tostring() return 'Tanh()' end

-- gelu activation
local GELU = {}
GELU.__index = GELU

function GELU.new()
    return setmetatable({}, GELU)
end

function GELU:forward(input)
    local out = Tensor.gelu(input)

    autograd.record("gelu", {input}, out, function(grad)
        if input.requires_grad then
            -- approximate gelu gradient
            -- using finite difference style: store input for backward
            -- d(gelu)/dx ~ 0.5*(1+tanh(c*(x+0.044715*x^3))) + x*sech^2(...)*c*(1+3*0.044715*x^2)*0.5
            -- for simplicity we use the numerical gradient from the output
            -- gelu'(x) = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
            local sqrt_2_pi = 0.7978845608
            local n = input:numel()
            local dgelu = Tensor.new(input.shape)
            for i = 0, n - 1 do
                local x = input:get(i)
                local inner = sqrt_2_pi * (x + 0.044715 * x * x * x)
                local tanh_inner = math.tanh(inner)
                local sech2 = 1.0 - tanh_inner * tanh_inner
                local d = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x * x)
                dgelu:set(i, d)
            end
            autograd.acc_grad(input, Tensor.mul(grad, dgelu))
        end
    end)

    return out
end

GELU.__call = function(self, input)
    return self:forward(input)
end

function GELU:parameters() return {} end
function GELU:zero_grad() end
function GELU:__tostring() return 'GELU()' end

-- silu activation
local SiLU = {}
SiLU.__index = SiLU

function SiLU.new()
    return setmetatable({}, SiLU)
end

function SiLU:forward(input)
    local out = Tensor.silu(input)

    autograd.record("silu", {input}, out, function(grad)
        if input.requires_grad then
            -- d(silu)/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            -- = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            local sig = Tensor.sigmoid(input)
            local ones = Tensor.new(input.shape)
            ones:ones()
            local one_minus_sig = Tensor.sub(ones, sig)
            local x_term = Tensor.mul(input, one_minus_sig)
            local inner = Tensor.add(ones, x_term)
            local dsilu = Tensor.mul(sig, inner)
            autograd.acc_grad(input, Tensor.mul(grad, dsilu))
        end
    end)

    return out
end

SiLU.__call = function(self, input)
    return self:forward(input)
end

function SiLU:parameters() return {} end
function SiLU:zero_grad() end
function SiLU:__tostring() return 'SiLU()' end

return {
    ReLU    = ReLU,
    Sigmoid = Sigmoid,
    Tanh    = Tanh,
    GELU    = GELU,
    SiLU    = SiLU,
}
