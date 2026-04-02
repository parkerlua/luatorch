local Tensor             = require('luatorch.tensor')
local autograd           = require('luatorch.autograd')
local Linear             = require('luatorch.nn.linear')
local LayerNorm          = require('luatorch.nn.layernorm')
local Dropout            = require('luatorch.nn.dropout')
local MultiHeadAttention = require('luatorch.nn.multihead')
local activation         = require('luatorch.nn.activation')

-- try to load flash attention, falls back to naive if not available
local ok, FlashAttention = pcall(require, 'luatorch.nn.flash_attention')
if not ok then FlashAttention = nil end

-- transformer block
-- the core repeating unit of GPT, BERT, and every modern language model
-- pre-norm style: normalize before attention and FFN, not after
-- this is more stable for training deep networks
-- each block has:
--   layernorm -> multi head attention -> residual add
--   layernorm -> feedforward network -> residual add

local TransformerBlock = {}
TransformerBlock.__index = TransformerBlock

-- embed_dim  = model dimension (e.g. 768)
-- num_heads  = attention heads (e.g. 12)
-- dropout_p  = dropout probability, default 0.1
-- ffn_mult   = FFN hidden dim multiplier, default 4
-- use_flash  = use flash attention when on CUDA, default true
function TransformerBlock.new(embed_dim, num_heads, dropout_p, ffn_mult, use_flash)
    local self = setmetatable({}, TransformerBlock)

    dropout_p = dropout_p or 0.1
    ffn_mult  = ffn_mult  or 4
    if use_flash == nil then use_flash = true end

    self.embed_dim = embed_dim
    local ffn_dim  = embed_dim * ffn_mult

    -- attention sublayer
    -- use flash attention if available and requested
    self.ln1 = LayerNorm.new(embed_dim)
    if use_flash and FlashAttention then
        self.attn = FlashAttention.new(embed_dim, num_heads, true)
    else
        self.attn = MultiHeadAttention.new(embed_dim, num_heads)
    end
    self.drop1 = Dropout.new(dropout_p)

    -- feedforward sublayer
    -- two linear layers with GELU between
    -- expands to 4x embed_dim then projects back down
    self.ln2   = LayerNorm.new(embed_dim)
    self.ffn1  = Linear.new(embed_dim, ffn_dim)
    self.gelu  = activation.GELU.new()
    self.ffn2  = Linear.new(ffn_dim, embed_dim)
    self.drop2 = Dropout.new(dropout_p)

    return self
end

-- forward pass
-- x shape is [batch * seq_len, embed_dim] (flattened for 2D matmuls)
-- mask is optional, used for causal attention
function TransformerBlock:forward(x)
    -- attention with residual
    -- pre-norm: normalize before attention
    local normed = self.ln1(x)
    local attn_out = self.attn(normed)
    attn_out = self.drop1(attn_out)
    x = Tensor.add(x, attn_out)  -- residual connection

    -- feedforward with residual
    -- pre-norm: normalize before FFN
    normed = self.ln2(x)
    local ffn_out = self.ffn1(normed)
    ffn_out = self.gelu(ffn_out)
    ffn_out = self.ffn2(ffn_out)
    ffn_out = self.drop2(ffn_out)
    x = Tensor.add(x, ffn_out)  -- residual connection

    return x
end

TransformerBlock.__call = function(self, x)
    return self:forward(x)
end

function TransformerBlock:parameters()
    local params = {}
    local sublayers = {
        self.ln1, self.attn, self.ln2,
        self.ffn1, self.ffn2
    }
    for _, layer in ipairs(sublayers) do
        if layer.parameters then
            for _, p in ipairs(layer:parameters()) do
                table.insert(params, p)
            end
        end
    end
    return params
end

function TransformerBlock:zero_grad()
    autograd.zero_grad(self:parameters())
end

function TransformerBlock:num_params()
    local n = 0
    local sublayers = {
        self.ln1, self.attn, self.ln2,
        self.ffn1, self.ffn2
    }
    for _, layer in ipairs(sublayers) do
        if layer.num_params then
            n = n + layer:num_params()
        end
    end
    return n
end

-- set training or eval mode (controls dropout)
function TransformerBlock:train()
    self.drop1:train()
    self.drop2:train()
end

function TransformerBlock:eval()
    self.drop1:eval()
    self.drop2:eval()
end

function TransformerBlock:__tostring()
    return string.format('TransformerBlock(embed=%d, params=%d)',
        self.embed_dim, self:num_params())
end

return TransformerBlock
