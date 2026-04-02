local Tensor           = require('luatorch.tensor')
local autograd         = require('luatorch.autograd')
local Linear           = require('luatorch.nn.linear')
local Embedding        = require('luatorch.nn.embedding')
local LayerNorm        = require('luatorch.nn.layernorm')
local Dropout          = require('luatorch.nn.dropout')
local TransformerBlock = require('luatorch.nn.transformer')
local positional       = require('luatorch.nn.positional')

-- GPT style language model
-- embedding + positional encoding + N transformer blocks + output head
-- predicts the next token given previous tokens
-- causal masking prevents attending to future positions

local GPT = {}
GPT.__index = GPT

-- vocab_size   = number of unique tokens
-- embed_dim    = model dimension
-- num_heads    = attention heads per block
-- num_layers   = number of transformer blocks
-- max_seq_len  = maximum sequence length
-- dropout      = dropout probability, default 0.1
function GPT.new(vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout)
    local self = setmetatable({}, GPT)

    dropout = dropout or 0.1

    self.vocab_size  = vocab_size
    self.embed_dim   = embed_dim
    self.num_heads   = num_heads
    self.num_layers  = num_layers
    self.max_seq_len = max_seq_len

    -- token embedding
    self.token_emb = Embedding.new(vocab_size, embed_dim)

    -- positional encoding
    self.pos_enc = positional.SinusoidalPE.new(max_seq_len, embed_dim)

    -- embedding dropout
    self.emb_drop = Dropout.new(dropout)

    -- transformer blocks
    self.blocks = {}
    for i = 1, num_layers do
        self.blocks[i] = TransformerBlock.new(embed_dim, num_heads, dropout)
    end

    -- final layer norm
    self.final_ln = LayerNorm.new(embed_dim)

    -- output head projects from embed_dim to vocab_size
    self.head = Linear.new(embed_dim, vocab_size, false)

    return self
end

-- forward pass
-- input is [batch, seq_len] tensor of token ids
-- returns [batch * seq_len, vocab_size] logits
function GPT:forward(input)
    local batch   = input.shape[1]
    local seq_len = input.shape[2]

    -- flatten input to [batch * seq_len] for embedding lookup
    local flat_input = Tensor.new({batch * seq_len})
    for i = 0, batch * seq_len - 1 do
        flat_input:set(i, input:get(i))
    end

    -- token embeddings: [batch * seq_len, embed_dim]
    local x = self.token_emb(flat_input)

    -- add positional encoding
    x = self.pos_enc(x, seq_len)

    -- embedding dropout
    x = self.emb_drop(x)

    -- pass through transformer blocks
    for _, block in ipairs(self.blocks) do
        x = block(x)
    end

    -- final layer norm
    x = self.final_ln(x)

    -- project to vocab size
    local logits = self.head(x)

    return logits
end

GPT.__call = function(self, input)
    return self:forward(input)
end

-- generate text token by token
-- prompt = tensor of token ids [1, prompt_len]
-- max_tokens = how many new tokens to generate
-- temperature = controls randomness, lower = more deterministic
-- top_k = only sample from top k most likely tokens
function GPT:generate(prompt, max_tokens, temperature, top_k)
    temperature = temperature or 1.0
    top_k       = top_k       or 0
    max_tokens  = max_tokens  or 100

    -- disable dropout and gradient tracking
    self:eval()
    autograd.enabled = false

    -- start with the prompt tokens as a lua table
    local tokens = {}
    for i = 0, prompt:numel() - 1 do
        table.insert(tokens, math.floor(prompt:get(i)))
    end

    for _ = 1, max_tokens do
        -- take last max_seq_len tokens
        local ctx_len = math.min(#tokens, self.max_seq_len)
        local start   = #tokens - ctx_len + 1

        -- build input tensor [1, ctx_len]
        local input = Tensor.new({1, ctx_len})
        for i = 0, ctx_len - 1 do
            input:set(i, tokens[start + i])
        end

        -- forward pass
        autograd.zero_graph()
        local logits = self(input)

        -- get logits for the last position
        -- logits shape is [ctx_len, vocab_size]
        local last_logits = {}
        local offset = (ctx_len - 1) * self.vocab_size
        for v = 0, self.vocab_size - 1 do
            last_logits[v] = logits:get(offset + v)
        end

        -- apply temperature
        if temperature ~= 1.0 then
            for v = 0, self.vocab_size - 1 do
                last_logits[v] = last_logits[v] / temperature
            end
        end

        -- top-k filtering
        if top_k > 0 and top_k < self.vocab_size then
            -- find kth largest value
            local sorted = {}
            for v = 0, self.vocab_size - 1 do
                table.insert(sorted, last_logits[v])
            end
            table.sort(sorted, function(a, b) return a > b end)
            local threshold = sorted[top_k]

            -- zero out everything below threshold
            for v = 0, self.vocab_size - 1 do
                if last_logits[v] < threshold then
                    last_logits[v] = -math.huge
                end
            end
        end

        -- softmax
        local max_val = last_logits[0]
        for v = 1, self.vocab_size - 1 do
            if last_logits[v] > max_val then max_val = last_logits[v] end
        end

        local sum = 0.0
        local probs = {}
        for v = 0, self.vocab_size - 1 do
            probs[v] = math.exp(last_logits[v] - max_val)
            sum = sum + probs[v]
        end
        for v = 0, self.vocab_size - 1 do
            probs[v] = probs[v] / sum
        end

        -- sample from the distribution
        local r = math.random()
        local cumsum = 0.0
        local next_token = 0
        for v = 0, self.vocab_size - 1 do
            cumsum = cumsum + probs[v]
            if r <= cumsum then
                next_token = v
                break
            end
        end

        table.insert(tokens, next_token)
    end

    autograd.enabled = true
    self:train()

    -- return as tensor
    local result = Tensor.new({#tokens})
    for i, t in ipairs(tokens) do
        result:set(i - 1, t)
    end
    return result
end

function GPT:parameters()
    local params = {}

    -- token embedding
    for _, p in ipairs(self.token_emb:parameters()) do
        table.insert(params, p)
    end

    -- positional encoding (sinusoidal has none, learned has some)
    if self.pos_enc.parameters then
        for _, p in ipairs(self.pos_enc:parameters()) do
            table.insert(params, p)
        end
    end

    -- transformer blocks
    for _, block in ipairs(self.blocks) do
        for _, p in ipairs(block:parameters()) do
            table.insert(params, p)
        end
    end

    -- final layer norm
    for _, p in ipairs(self.final_ln:parameters()) do
        table.insert(params, p)
    end

    -- output head
    for _, p in ipairs(self.head:parameters()) do
        table.insert(params, p)
    end

    return params
end

function GPT:zero_grad()
    autograd.zero_grad(self:parameters())
end

function GPT:num_params()
    local n = 0
    for _, _ in ipairs(self:parameters()) do
        n = n + 1
    end
    -- count actual elements
    local total = 0
    for _, p in ipairs(self:parameters()) do
        total = total + p:numel()
    end
    return total
end

-- switch between training and eval mode
function GPT:train()
    self.emb_drop:train()
    for _, block in ipairs(self.blocks) do
        block:train()
    end
end

function GPT:eval()
    self.emb_drop:eval()
    for _, block in ipairs(self.blocks) do
        block:eval()
    end
end

-- collect parameters that should skip weight decay
-- bias and layernorm parameters dont benefit from weight decay
function GPT:no_decay_params()
    local no_decay = {}

    -- layernorm gamma and beta
    for _, p in ipairs(self.final_ln:parameters()) do
        table.insert(no_decay, p)
    end
    for _, block in ipairs(self.blocks) do
        for _, p in ipairs(block.ln1:parameters()) do
            table.insert(no_decay, p)
        end
        for _, p in ipairs(block.ln2:parameters()) do
            table.insert(no_decay, p)
        end
        -- bias parameters from linear layers
        if block.ffn1.use_bias then
            table.insert(no_decay, block.ffn1.bias)
        end
        if block.ffn2.use_bias then
            table.insert(no_decay, block.ffn2.bias)
        end
        if block.attn.out_proj.use_bias then
            table.insert(no_decay, block.attn.out_proj.bias)
        end
    end

    return no_decay
end

function GPT:__tostring()
    local parts = {string.format('GPT(vocab=%d, embed=%d, heads=%d, layers=%d, max_seq=%d)',
        self.vocab_size, self.embed_dim, self.num_heads,
        self.num_layers, self.max_seq_len)}
    table.insert(parts, string.format('  %s', tostring(self.token_emb)))
    table.insert(parts, string.format('  %s', tostring(self.pos_enc)))
    for i, block in ipairs(self.blocks) do
        table.insert(parts, string.format('  [%d] %s', i, tostring(block)))
    end
    table.insert(parts, string.format('  %s', tostring(self.final_ln)))
    table.insert(parts, string.format('  %s', tostring(self.head)))
    table.insert(parts, string.format('  total params: %d', self:num_params()))
    return table.concat(parts, '\n')
end

return GPT
