local activation = require('luatorch.nn.activation')
local batchnorm  = require('luatorch.nn.batchnorm')
local positional = require('luatorch.nn.positional')
local loss_mod   = require('luatorch.nn.loss')

-- make modules callable as Module(...) instead of Module.new(...)
local function callable(mod)
    return setmetatable({}, {
        __index = mod,
        __call  = function(_, ...) return mod.new(...) end,
    })
end

-- try loading flash attention, not fatal if missing
local ok_flash, FlashAttn = pcall(require, 'luatorch.nn.flash_attention')

return {
    -- layers
    Linear             = callable(require('luatorch.nn.linear')),
    Conv2d             = callable(require('luatorch.nn.conv2d')),
    Embedding          = callable(require('luatorch.nn.embedding')),
    Dropout            = callable(require('luatorch.nn.dropout')),
    LayerNorm          = callable(require('luatorch.nn.layernorm')),
    BatchNorm1d        = callable(batchnorm.BatchNorm1d),
    Sequential         = callable(require('luatorch.nn.sequential')),
    MultiHeadAttention = callable(require('luatorch.nn.multihead')),
    FlashMultiHeadAttention = ok_flash and callable(FlashAttn) or nil,
    TransformerBlock   = callable(require('luatorch.nn.transformer')),

    -- activations
    ReLU    = callable(activation.ReLU),
    Sigmoid = callable(activation.Sigmoid),
    Tanh    = callable(activation.Tanh),
    GELU    = callable(activation.GELU),
    SiLU    = callable(activation.SiLU),

    -- positional encodings
    SinusoidalPE = callable(positional.SinusoidalPE),
    LearnedPE    = callable(positional.LearnedPE),

    -- loss functions
    MSELoss          = callable(loss_mod.MSELoss),
    MAELoss          = callable(loss_mod.MAELoss),
    CrossEntropyLoss = callable(loss_mod.CrossEntropyLoss),
}
