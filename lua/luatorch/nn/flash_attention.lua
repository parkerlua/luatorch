local Tensor   = require('luatorch.tensor')
local autograd = require('luatorch.autograd')
local Linear   = require('luatorch.nn.linear')

-- flash attention multi head attention
-- uses the flash attention kernel on CUDA to avoid materializing N*N matrix
-- falls back to naive attention on CPU
-- drop-in replacement for MultiHeadAttention

local FlashMultiHeadAttention = {}
FlashMultiHeadAttention.__index = FlashMultiHeadAttention

function FlashMultiHeadAttention.new(embed_dim, num_heads, causal)
    local self = setmetatable({}, FlashMultiHeadAttention)

    assert(embed_dim % num_heads == 0,
        'luatorch error: embed_dim must be divisible by num_heads')

    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim  = embed_dim / num_heads
    self.causal    = causal or false

    self.q_proj   = Linear.new(embed_dim, embed_dim, false)
    self.k_proj   = Linear.new(embed_dim, embed_dim, false)
    self.v_proj   = Linear.new(embed_dim, embed_dim, false)
    self.out_proj = Linear.new(embed_dim, embed_dim)

    self.scale = 1.0 / math.sqrt(self.head_dim)

    return self
end

-- fix: seq_len and batch_size are passed explicitly instead of hidden _seq_len field
-- caller sets these before calling forward, or they default to total_tokens and 1
function FlashMultiHeadAttention:forward(query, key, value, seq_len)
    key   = key   or query
    value = value or query

    -- query is [batch * seq_len, embed_dim] (flattened)
    local total_tokens = query.shape[1]
    local embed = self.embed_dim
    local heads = self.num_heads
    local hdim  = self.head_dim

    -- project q, k, v
    local q = self.q_proj(query)
    local k = self.k_proj(key)
    local v = self.v_proj(value)

    -- try flash attention on CUDA
    if query.device == 'cuda' then
        -- fix: use explicit seq_len parameter instead of undocumented _seq_len field
        seq_len = seq_len or total_tokens
        local batch = total_tokens / seq_len

        -- rearrange [batch*seq, embed] to [batch*heads, seq, hdim]
        -- this is done by treating the embed dimension as [heads, hdim]
        local q_flash = Tensor.new({batch * heads, seq_len, hdim})
        local k_flash = Tensor.new({batch * heads, seq_len, hdim})
        local v_flash = Tensor.new({batch * heads, seq_len, hdim})

        -- copy with reordering: for each token, scatter heads
        for b = 0, batch - 1 do
            for s = 0, seq_len - 1 do
                local token_idx = b * seq_len + s
                for h = 0, heads - 1 do
                    for d = 0, hdim - 1 do
                        local src = token_idx * embed + h * hdim + d
                        local dst = (b * heads + h) * seq_len * hdim + s * hdim + d
                        q_flash:set(dst, q:get(src))
                        k_flash:set(dst, k:get(src))
                        v_flash:set(dst, v:get(src))
                    end
                end
            end
        end

        -- move to cuda for flash attention
        q_flash:cuda()
        k_flash:cuda()
        v_flash:cuda()

        local attn_out = Tensor.flash_attention(
            q_flash, k_flash, v_flash,
            batch * heads, seq_len, hdim,
            self.causal)

        if attn_out then
            -- rearrange back: [batch*heads, seq, hdim] -> [batch*seq, embed]
            attn_out:cpu()
            local concat = Tensor.new({total_tokens, embed})
            for b = 0, batch - 1 do
                for s = 0, seq_len - 1 do
                    local token_idx = b * seq_len + s
                    for h = 0, heads - 1 do
                        for d = 0, hdim - 1 do
                            local src = (b * heads + h) * seq_len * hdim + s * hdim + d
                            local dst = token_idx * embed + h * hdim + d
                            concat:set(dst, attn_out:get(src))
                        end
                    end
                end
            end

            local out = self.out_proj(concat)
            return out
        end
    end

    -- fallback: naive attention (same as MultiHeadAttention)
    local out_parts = {}
    for h = 0, heads - 1 do
        local q_head = Tensor.new({total_tokens, hdim})
        local k_head = Tensor.new({total_tokens, hdim})
        local v_head = Tensor.new({total_tokens, hdim})

        for i = 0, total_tokens - 1 do
            for d = 0, hdim - 1 do
                q_head:set(i * hdim + d, q:get(i * embed + h * hdim + d))
                k_head:set(i * hdim + d, k:get(i * embed + h * hdim + d))
                v_head:set(i * hdim + d, v:get(i * embed + h * hdim + d))
            end
        end

        local kt     = Tensor.transpose(k_head)
        local scores = autograd.matmul(q_head, kt)
        scores       = Tensor.mul_scalar(scores, self.scale)
        local attn   = Tensor.softmax(scores)
        out_parts[h] = autograd.matmul(attn, v_head)
    end

    local concat = Tensor.new({total_tokens, embed})
    for h = 0, heads - 1 do
        for i = 0, total_tokens - 1 do
            for d = 0, hdim - 1 do
                concat:set(i * embed + h * hdim + d,
                          out_parts[h]:get(i * hdim + d))
            end
        end
    end

    return self.out_proj(concat)
end

FlashMultiHeadAttention.__call = function(self, query, key, value)
    return self:forward(query, key, value)
end

function FlashMultiHeadAttention:parameters()
    local params = {}
    for _, layer in ipairs({self.q_proj, self.k_proj, self.v_proj, self.out_proj}) do
        for _, p in ipairs(layer:parameters()) do
            table.insert(params, p)
        end
    end
    return params
end

function FlashMultiHeadAttention:zero_grad()
    autograd.zero_grad(self:parameters())
end

function FlashMultiHeadAttention:num_params()
    local n = 0
    for _, layer in ipairs({self.q_proj, self.k_proj, self.v_proj, self.out_proj}) do
        n = n + layer:num_params()
    end
    return n
end

function FlashMultiHeadAttention:__tostring()
    return string.format('FlashMultiHeadAttention(embed=%d, heads=%d, causal=%s)',
        self.embed_dim, self.num_heads, tostring(self.causal))
end

return FlashMultiHeadAttention
