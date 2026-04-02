local Tensor   = require('luatorch.tensor')
local autograd = require('luatorch.autograd')
local Linear   = require('luatorch.nn.linear')

-- multi head attention
-- the core mechanism behind transformers
-- splits the input into multiple heads so the model can attend
-- to different parts of the sequence in different ways
-- each head learns its own attention pattern

local MultiHeadAttention = {}
MultiHeadAttention.__index = MultiHeadAttention

-- embed_dim  = total dimension of the model (e.g. 768)
-- num_heads  = number of attention heads (e.g. 12)
-- embed_dim must be divisible by num_heads
function MultiHeadAttention.new(embed_dim, num_heads)
    local self = setmetatable({}, MultiHeadAttention)

    assert(embed_dim % num_heads == 0,
        'luatorch error: embed_dim must be divisible by num_heads')

    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim  = embed_dim / num_heads

    -- projections for query, key, value
    -- each one maps [batch, seq, embed_dim] -> [batch, seq, embed_dim]
    self.q_proj   = Linear.new(embed_dim, embed_dim, false)
    self.k_proj   = Linear.new(embed_dim, embed_dim, false)
    self.v_proj   = Linear.new(embed_dim, embed_dim, false)
    self.out_proj = Linear.new(embed_dim, embed_dim)

    -- scale factor for dot product attention
    -- dividing by sqrt(head_dim) keeps the dot products from getting too large
    -- which would make softmax saturate and kill gradients
    self.scale = 1.0 / math.sqrt(self.head_dim)

    return self
end

-- forward pass
-- input shape is [batch, seq_len, embed_dim]
-- for self attention, q k v are all the same input
-- returns [batch, seq_len, embed_dim]
function MultiHeadAttention:forward(query, key, value)
    -- default to self attention
    key   = key   or query
    value = value or query

    local batch   = query.shape[1]
    local seq_len = query.shape[2]
    local heads   = self.num_heads
    local hdim    = self.head_dim

    -- project q, k, v
    -- reshape from [batch, seq, embed] to [batch*seq, embed] for the linear layers
    local q_flat = query
    local k_flat = key
    local v_flat = value

    -- if 3D, reshape to 2D for matmul in linear layers
    if query.ndim == 3 then
        q_flat = Tensor.new({batch * seq_len, self.embed_dim})
        k_flat = Tensor.new({batch * seq_len, self.embed_dim})
        v_flat = Tensor.new({batch * seq_len, self.embed_dim})
        for i = 0, batch * seq_len * self.embed_dim - 1 do
            q_flat:set(i, query:get(i))
            k_flat:set(i, key:get(i))
            v_flat:set(i, value:get(i))
        end
    end

    local q = self.q_proj(q_flat)  -- [batch*seq, embed]
    local k = self.k_proj(k_flat)
    local v = self.v_proj(v_flat)

    -- compute attention scores for each head
    -- we process one head at a time to stay with 2D matmuls
    -- q_head is [batch*seq, head_dim] for one head
    local out_parts = {}

    for h = 0, heads - 1 do
        -- extract this head's slice from q, k, v
        local q_head = Tensor.new({batch * seq_len, hdim})
        local k_head = Tensor.new({batch * seq_len, hdim})
        local v_head = Tensor.new({batch * seq_len, hdim})

        for i = 0, batch * seq_len - 1 do
            for d = 0, hdim - 1 do
                q_head:set(i * hdim + d, q:get(i * self.embed_dim + h * hdim + d))
                k_head:set(i * hdim + d, k:get(i * self.embed_dim + h * hdim + d))
                v_head:set(i * hdim + d, v:get(i * self.embed_dim + h * hdim + d))
            end
        end

        -- for each batch item, compute attention
        -- scores = q_head @ k_head^T * scale  -> [seq, seq]
        -- attn   = softmax(scores)
        -- head_out = attn @ v_head             -> [seq, hdim]
        local head_out = Tensor.new({batch * seq_len, hdim})

        for b = 0, batch - 1 do
            -- extract [seq, hdim] for this batch item
            local qb = Tensor.new({seq_len, hdim})
            local kb = Tensor.new({seq_len, hdim})
            local vb = Tensor.new({seq_len, hdim})

            for s = 0, seq_len - 1 do
                for d = 0, hdim - 1 do
                    local idx = (b * seq_len + s) * hdim + d
                    qb:set(s * hdim + d, q_head:get(idx))
                    kb:set(s * hdim + d, k_head:get(idx))
                    vb:set(s * hdim + d, v_head:get(idx))
                end
            end

            -- scores = Q @ K^T
            local kt     = Tensor.transpose(kb)
            local scores = autograd.matmul(qb, kt)

            -- scale scores
            scores = Tensor.mul_scalar(scores, self.scale)

            -- softmax along last dim
            local attn = Tensor.softmax(scores)

            -- weighted sum of values
            local head_b = autograd.matmul(attn, vb)

            -- copy back into head_out
            for s = 0, seq_len - 1 do
                for d = 0, hdim - 1 do
                    head_out:set((b * seq_len + s) * hdim + d,
                                head_b:get(s * hdim + d))
                end
            end
        end

        out_parts[h] = head_out
    end

    -- concatenate all heads back together -> [batch*seq, embed_dim]
    local concat = Tensor.new({batch * seq_len, self.embed_dim})
    for h = 0, heads - 1 do
        for i = 0, batch * seq_len - 1 do
            for d = 0, hdim - 1 do
                concat:set(i * self.embed_dim + h * hdim + d,
                          out_parts[h]:get(i * hdim + d))
            end
        end
    end

    -- final output projection
    local out = self.out_proj(concat)

    return out
end

MultiHeadAttention.__call = function(self, query, key, value)
    return self:forward(query, key, value)
end

function MultiHeadAttention:parameters()
    local params = {}
    for _, layer in ipairs({self.q_proj, self.k_proj, self.v_proj, self.out_proj}) do
        for _, p in ipairs(layer:parameters()) do
            table.insert(params, p)
        end
    end
    return params
end

function MultiHeadAttention:zero_grad()
    autograd.zero_grad(self:parameters())
end

function MultiHeadAttention:num_params()
    local n = 0
    for _, layer in ipairs({self.q_proj, self.k_proj, self.v_proj, self.out_proj}) do
        n = n + layer:num_params()
    end
    return n
end

function MultiHeadAttention:__tostring()
    return string.format('MultiHeadAttention(embed=%d, heads=%d, head_dim=%d, params=%d)',
        self.embed_dim, self.num_heads, self.head_dim, self:num_params())
end

return MultiHeadAttention
