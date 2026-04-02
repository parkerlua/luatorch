local Tensor     = require('luatorch.tensor')
local autograd   = require('luatorch.autograd')
local Linear     = require('luatorch.nn.linear')
local Sequential = require('luatorch.nn.sequential')
local LayerNorm  = require('luatorch.nn.layernorm')
local Dropout    = require('luatorch.nn.dropout')
local Embedding  = require('luatorch.nn.embedding')
local activation = require('luatorch.nn.activation')

local function assert_near(a, b, tol, msg)
    tol = tol or 1e-3
    if math.abs(a - b) > tol then
        error(string.format('%s: expected ~%f got %f', msg or 'assert_near', b, a))
    end
end

-- Linear forward produces correct output shape
do
    local layer = Linear.new(4, 3)
    local input = Tensor.new({2, 4}); input:rand()
    autograd.zero_graph()
    local out = layer(input)
    assert(out.shape[1] == 2 and out.shape[2] == 3,
        string.format('linear output shape: expected 2x3 got %dx%d', out.shape[1], out.shape[2]))
end

-- Linear parameters returns weight and bias
do
    local layer = Linear.new(4, 3)
    local params = layer:parameters()
    assert(#params == 2, 'linear should have 2 params (weight + bias)')

    local no_bias = Linear.new(4, 3, false)
    params = no_bias:parameters()
    assert(#params == 1, 'linear without bias should have 1 param')
end

-- Sequential chains correctly
do
    autograd.zero_graph()
    local model = Sequential.new(
        Linear.new(4, 3),
        activation.ReLU.new(),
        Linear.new(3, 2)
    )
    local input = Tensor.new({2, 4}); input:rand()
    local out = model(input)
    assert(out.shape[1] == 2 and out.shape[2] == 2,
        'sequential output shape should be 2x2')

    local params = model:parameters()
    assert(#params > 0, 'sequential should have parameters')
end

-- LayerNorm output has mean near 0 and std near 1
do
    autograd.zero_graph()
    local ln = LayerNorm.new(8)
    local input = Tensor.new({4, 8}); input:randn()
    local out = ln(input)

    -- check each row has mean ~0 and variance ~1
    for r = 0, 3 do
        local row_sum = 0
        for c = 0, 7 do
            row_sum = row_sum + out:get(r * 8 + c)
        end
        local row_mean = row_sum / 8

        local var_sum = 0
        for c = 0, 7 do
            local diff = out:get(r * 8 + c) - row_mean
            var_sum = var_sum + diff * diff
        end
        local row_var = var_sum / 8

        assert_near(row_mean, 0.0, 0.01, 'layernorm mean')
        assert_near(row_var, 1.0, 0.1, 'layernorm variance')
    end
end

-- Dropout zeros approximately p fraction in train mode
do
    autograd.zero_graph()
    local drop = Dropout.new(0.5)
    local input = Tensor.new({1000}); input:ones()

    local out = drop(input)
    local n_zeros = 0
    for i = 0, 999 do
        if out:get(i) == 0.0 then n_zeros = n_zeros + 1 end
    end

    -- expect roughly 50% zeros with some tolerance
    local drop_rate = n_zeros / 1000
    assert(drop_rate > 0.3 and drop_rate < 0.7,
        string.format('dropout rate should be ~0.5, got %.2f', drop_rate))
end

-- Dropout passes all elements in eval mode
do
    autograd.zero_graph()
    local drop = Dropout.new(0.5)
    drop:eval()
    local input = Tensor.new({100}); input:ones()
    local out = drop(input)
    for i = 0, 99 do
        assert_near(out:get(i), 1.0, 1e-5, 'dropout eval should pass through')
    end
end

-- Embedding lookup returns correct rows
do
    autograd.zero_graph()
    local emb = Embedding.new(10, 4)
    -- set known values for token 3
    for d = 0, 3 do
        emb.weight:set(3 * 4 + d, (d + 1) * 1.0)
    end

    local input = Tensor.new({1}); input:set(0, 3)
    local out = emb(input)
    assert_near(out:get(0), 1.0, 1e-5, 'emb lookup [0]')
    assert_near(out:get(1), 2.0, 1e-5, 'emb lookup [1]')
    assert_near(out:get(2), 3.0, 1e-5, 'emb lookup [2]')
    assert_near(out:get(3), 4.0, 1e-5, 'emb lookup [3]')
end
