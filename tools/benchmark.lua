-- luatorch benchmark
-- tests every major operation at multiple sizes
-- reports timing, gflops, and memory usage

local lf = require('luatorch')

local Tensor      = lf.Tensor
local autograd    = lf.autograd
local nn          = lf.nn
local Linear      = require('luatorch.nn.linear')
local activation  = require('luatorch.nn.activation')

-- timing helper
local function time_it(name, fn, warmup, iters)
    warmup = warmup or 3
    iters  = iters  or 10

    -- warmup
    for _ = 1, warmup do fn() end

    local start = os.clock()
    for _ = 1, iters do fn() end
    local elapsed = os.clock() - start

    local ms = (elapsed / iters) * 1000
    return ms
end

-- results table
local results = {}
local function record(op, size, device, time_ms, gflops)
    table.insert(results, {
        op      = op,
        size    = size,
        device  = device,
        time_ms = time_ms,
        gflops  = gflops or 0,
    })
end

-- print results as a table
local function print_results()
    print(string.format('\n%-35s  %-8s  %-8s  %10s  %10s',
        'operation', 'size', 'device', 'time(ms)', 'GFLOPS'))
    print(string.rep('-', 80))
    for _, r in ipairs(results) do
        print(string.format('%-35s  %-8s  %-8s  %10.3f  %10.1f',
            r.op, r.size, r.device, r.time_ms, r.gflops))
    end
end

-- save to csv
local function save_csv(path)
    local f = io.open(path, 'w')
    f:write('operation,size,device,time_ms,gflops\n')
    for _, r in ipairs(results) do
        f:write(string.format('%s,%s,%s,%.4f,%.2f\n',
            r.op, r.size, r.device, r.time_ms, r.gflops))
    end
    f:close()
end

print('luatorch benchmark')
print(string.format('luajit: %s', jit and jit.version or 'unknown'))
print('')

-- tensor creation
for _, n in ipairs({256, 1024, 4096}) do
    local ms = time_it('tensor_new', function()
        local t = Tensor.new({n, n})
    end)
    record('tensor creation', n .. 'x' .. n, 'cpu', ms)
end

-- elementwise add
for _, n in ipairs({256, 1024, 4096}) do
    local a = Tensor.new({n, n}); a:rand()
    local b = Tensor.new({n, n}); b:rand()

    local ms = time_it('add', function()
        Tensor.add(a, b)
    end)

    -- 1 flop per element
    local flops = n * n
    local gflops = (flops / (ms / 1000)) / 1e9
    record('elementwise add', n .. 'x' .. n, 'cpu', ms, gflops)
end

-- matmul
for _, n in ipairs({256, 1024, 4096}) do
    local a = Tensor.new({n, n}); a:rand()
    local b = Tensor.new({n, n}); b:rand()

    local iters = n <= 1024 and 10 or 3
    local ms = time_it('matmul', function()
        Tensor.matmul(a, b)
    end, 2, iters)

    -- 2*n^3 flops for matmul
    local flops = 2.0 * n * n * n
    local gflops = (flops / (ms / 1000)) / 1e9
    record('matmul', n .. 'x' .. n, 'cpu', ms, gflops)
end

-- linear layer forward
for _, size in ipairs({{256, 256}, {1024, 1024}, {4096, 4096}}) do
    local layer = Linear.new(size[1], size[2])
    local input = Tensor.new({32, size[1]}); input:rand()

    local ms = time_it('linear forward', function()
        layer(input)
    end)

    local label = size[1] .. '->' .. size[2]
    record('linear forward (bs=32)', label, 'cpu', ms)
end

-- relu
for _, n in ipairs({256, 1024, 4096}) do
    local a = Tensor.new({n, n}); a:randn()
    local ms = time_it('relu', function()
        Tensor.relu(a)
    end)
    record('relu', n .. 'x' .. n, 'cpu', ms)
end

-- softmax
for _, n in ipairs({256, 1024, 4096}) do
    local a = Tensor.new({32, n}); a:randn()
    local ms = time_it('softmax', function()
        Tensor.softmax(a)
    end)
    record('softmax (bs=32)', tostring(n), 'cpu', ms)
end

-- full training step (forward + backward + optimizer)
for _, hidden in ipairs({128, 256, 512}) do
    local model = nn.Sequential(
        nn.Linear(128, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 10)
    )
    local params    = model:parameters()
    local optimizer = lf.optim.Adam.new(params, 0.001)
    local criterion = nn.CrossEntropyLoss()

    local input  = Tensor.new({32, 128}); input:rand()
    local target = Tensor.new({32})
    for i = 0, 31 do target:set(i, math.random(0, 9)) end

    local ms = time_it('train step', function()
        autograd.zero_graph()
        model:zero_grad()
        autograd.watch(input)
        local pred = model(input)
        local loss = criterion(pred, target)
        autograd.backward(loss)
        optimizer:step()
    end, 2, 5)

    record('training step (bs=32)', '128->' .. hidden .. '->10', 'cpu', ms)
end

-- gpt forward pass at different sequence lengths
for _, seq_len in ipairs({32, 64, 128}) do
    local gpt = lf.models.GPT.new(65, 128, 4, 2, 256, 0.0)
    local input = Tensor.new({1, seq_len})
    for i = 0, seq_len - 1 do input:set(i, math.random(0, 64)) end

    local ms = time_it('gpt forward', function()
        autograd.zero_graph()
        gpt(input)
    end, 1, 3)

    record('gpt forward (2L 4H 128d)', 'seq=' .. seq_len, 'cpu', ms)
end

-- gpt training step with tokens/sec
local gpt = lf.models.GPT.new(65, 128, 4, 2, 256, 0.0)
local gpt_params = gpt:parameters()
local gpt_opt    = lf.optim.Adam.new(gpt_params, 0.001)
local gpt_crit   = nn.CrossEntropyLoss()
local gpt_seq    = 64
local gpt_batch  = 2

local gpt_input  = Tensor.new({gpt_batch, gpt_seq})
local gpt_target = Tensor.new({gpt_batch * gpt_seq})
for i = 0, gpt_batch * gpt_seq - 1 do
    gpt_input:set(i, math.random(0, 64))
    gpt_target:set(i, math.random(0, 64))
end

local gpt_ms = time_it('gpt train', function()
    autograd.zero_graph()
    gpt:zero_grad()
    autograd.watch(gpt_input)
    local logits = gpt(gpt_input)
    local loss = gpt_crit(logits, gpt_target)
    autograd.backward(loss)
    gpt_opt:step()
end, 1, 3)

local tokens_per_step = gpt_batch * gpt_seq
local tokens_per_sec  = tokens_per_step / (gpt_ms / 1000)
record('gpt train step', 'bs=' .. gpt_batch .. ' seq=' .. gpt_seq, 'cpu', gpt_ms)

-- print results
print_results()

print(string.format('\ngpt training: %.0f tokens/sec (cpu, 2L 4H 128d bs=%d seq=%d)',
    tokens_per_sec, gpt_batch, gpt_seq))

-- memory
if Tensor.cuda_memory_allocated() > 0 then
    print(string.format('gpu memory: %.1f MB allocated, %.1f MB cached',
        Tensor.cuda_memory_allocated() / 1024 / 1024,
        Tensor.cuda_memory_cached() / 1024 / 1024))
end

-- save
save_csv('benchmark_results.csv')
print('\nresults saved to benchmark_results.csv')
