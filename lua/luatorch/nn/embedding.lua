local Tensor   = require('luatorch.tensor')
local autograd = require('luatorch.autograd')

-- embedding is a lookup table
-- converts integer token ids into dense vectors
-- every language model starts with one of these
-- vocab_size tokens each mapped to embedding_dim floats

local Embedding = {}
Embedding.__index = Embedding

-- vocab_size     = how many unique tokens (e.g. 50000 for GPT)
-- embedding_dim  = size of each vector (e.g. 768)
function Embedding.new(vocab_size, embedding_dim)
    local self = setmetatable({}, Embedding)

    self.vocab_size    = vocab_size
    self.embedding_dim = embedding_dim

    -- weight matrix is [vocab_size, embedding_dim]
    -- initialized from normal distribution
    self.weight = Tensor.new({vocab_size, embedding_dim})
    self.weight:randn()
    -- scale down so initial embeddings arent huge
    self.weight = Tensor.mul_scalar(self.weight, 0.02)
    autograd.watch(self.weight)

    return self
end

-- forward pass
-- input is a tensor of integer token ids, shape [batch] or [batch, seq_len]
-- output is [batch, embedding_dim] or [batch, seq_len, embedding_dim]
function Embedding:forward(input)
    local in_shape = input.shape
    local n_tokens = input:numel()
    local dim = self.embedding_dim

    -- figure out output shape
    local out_shape = {}
    for _, s in ipairs(in_shape) do
        table.insert(out_shape, s)
    end
    table.insert(out_shape, dim)

    local out = Tensor.new(out_shape)

    -- look up each token id and copy its embedding vector
    for i = 0, n_tokens - 1 do
        local token_id = math.floor(input:get(i))
        if token_id < 0 or token_id >= self.vocab_size then
            error(string.format('luatorch error: token id %d out of range [0, %d)',
                token_id, self.vocab_size))
        end

        -- copy embedding_dim floats from weight[token_id] to output
        for d = 0, dim - 1 do
            local val = self.weight:get(token_id * dim + d)
            out:set(i * dim + d, val)
        end
    end

    -- register backward
    -- gradient of embedding is sparse: only touched rows get updates
    autograd.record("embedding", {input}, out, function(grad)
        if self.weight.requires_grad then
            -- accumulate gradients into weight rows
            local dw = Tensor.new(self.weight.shape)
            dw:zeros()

            for i = 0, n_tokens - 1 do
                local token_id = math.floor(input:get(i))
                for d = 0, dim - 1 do
                    local old = dw:get(token_id * dim + d)
                    local g   = grad:get(i * dim + d)
                    dw:set(token_id * dim + d, old + g)
                end
            end

            autograd.acc_grad(self.weight, dw)
        end
    end)

    return out
end

Embedding.__call = function(self, input)
    return self:forward(input)
end

function Embedding:parameters()
    return {self.weight}
end

function Embedding:zero_grad()
    autograd.zero_grad(self:parameters())
end

function Embedding:num_params()
    return self.vocab_size * self.embedding_dim
end

function Embedding:__tostring()
    return string.format('Embedding(vocab=%d, dim=%d, params=%d)',
        self.vocab_size, self.embedding_dim, self:num_params())
end

return Embedding
