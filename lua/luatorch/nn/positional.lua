local Tensor   = require('luatorch.tensor')
local autograd = require('luatorch.autograd')

-- sinusoidal positional encoding
-- adds position information to token embeddings
-- without this, a transformer has no idea what order the tokens are in
-- uses sin for even dimensions, cos for odd dimensions
-- each dimension has a different frequency
-- no learnable parameters, just math

local SinusoidalPE = {}
SinusoidalPE.__index = SinusoidalPE

-- max_seq_len = maximum sequence length to support
-- embed_dim   = model dimension, must match embedding dim
function SinusoidalPE.new(max_seq_len, embed_dim)
    local self = setmetatable({}, SinusoidalPE)

    self.max_seq_len = max_seq_len
    self.embed_dim   = embed_dim

    -- precompute the encoding table
    -- shape [max_seq_len, embed_dim]
    self.pe = Tensor.new({max_seq_len, embed_dim})

    for pos = 0, max_seq_len - 1 do
        for i = 0, embed_dim - 1 do
            -- each pair of dimensions uses sin/cos at a different frequency
            -- the frequency decreases geometrically with dimension
            local div = math.exp(-(i - (i % 2)) * math.log(10000.0) / embed_dim)
            local angle = pos * div
            if i % 2 == 0 then
                self.pe:set(pos * embed_dim + i, math.sin(angle))
            else
                self.pe:set(pos * embed_dim + i, math.cos(angle))
            end
        end
    end

    return self
end

-- forward pass
-- input shape is [batch * seq_len, embed_dim]
-- we add positional encoding for the first seq_len positions
-- seq_len must be provided since input is flattened
function SinusoidalPE:forward(input, seq_len)
    local total = input:numel()
    local dim   = self.embed_dim
    local n_tokens = total / dim

    -- if seq_len not provided, assume input is [seq_len, embed_dim]
    seq_len = seq_len or n_tokens

    local out = Tensor.new(input.shape)

    for t = 0, n_tokens - 1 do
        local pos = t % seq_len  -- position within the sequence
        for d = 0, dim - 1 do
            local val = input:get(t * dim + d) + self.pe:get(pos * dim + d)
            out:set(t * dim + d, val)
        end
    end

    -- record for autograd, gradient just passes through
    autograd.record("positional", {input}, out, function(grad)
        if input.requires_grad then
            autograd.acc_grad(input, grad)
        end
    end)

    return out
end

SinusoidalPE.__call = function(self, input, seq_len)
    return self:forward(input, seq_len)
end

function SinusoidalPE:parameters() return {} end
function SinusoidalPE:zero_grad() end
function SinusoidalPE:num_params() return 0 end

function SinusoidalPE:__tostring()
    return string.format('SinusoidalPE(max_len=%d, dim=%d)',
        self.max_seq_len, self.embed_dim)
end

-- learned positional encoding
-- uses a learnable embedding table instead of fixed sin/cos
-- some models prefer this, slightly more flexible
local LearnedPE = {}
LearnedPE.__index = LearnedPE

function LearnedPE.new(max_seq_len, embed_dim)
    local self = setmetatable({}, LearnedPE)

    self.max_seq_len = max_seq_len
    self.embed_dim   = embed_dim

    -- learnable position embeddings
    self.weight = Tensor.new({max_seq_len, embed_dim})
    self.weight:randn()
    self.weight = Tensor.mul_scalar(self.weight, 0.02)
    autograd.watch(self.weight)

    return self
end

function LearnedPE:forward(input, seq_len)
    local total = input:numel()
    local dim   = self.embed_dim
    local n_tokens = total / dim

    seq_len = seq_len or n_tokens

    local out = Tensor.new(input.shape)

    for t = 0, n_tokens - 1 do
        local pos = t % seq_len
        for d = 0, dim - 1 do
            local val = input:get(t * dim + d) + self.weight:get(pos * dim + d)
            out:set(t * dim + d, val)
        end
    end

    autograd.record("learned_pe", {input}, out, function(grad)
        if input.requires_grad then
            autograd.acc_grad(input, grad)
        end
        if self.weight.requires_grad then
            local dw = Tensor.new(self.weight.shape)
            dw:zeros()
            for t = 0, n_tokens - 1 do
                local pos = t % seq_len
                for d = 0, dim - 1 do
                    local old = dw:get(pos * dim + d)
                    dw:set(pos * dim + d, old + grad:get(t * dim + d))
                end
            end
            autograd.acc_grad(self.weight, dw)
        end
    end)

    return out
end

LearnedPE.__call = function(self, input, seq_len)
    return self:forward(input, seq_len)
end

function LearnedPE:parameters() return {self.weight} end
function LearnedPE:zero_grad() autograd.zero_grad(self:parameters()) end
function LearnedPE:num_params() return self.max_seq_len * self.embed_dim end

function LearnedPE:__tostring()
    return string.format('LearnedPE(max_len=%d, dim=%d, params=%d)',
        self.max_seq_len, self.embed_dim, self:num_params())
end

return {
    SinusoidalPE = SinusoidalPE,
    LearnedPE    = LearnedPE,
}
