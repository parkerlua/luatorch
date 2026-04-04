-- benchmark_pytorch.lua
-- trains the same 2-16-1 XOR network used in examples/xor.lua
-- measures per-epoch time, total training time, final loss, and memory usage
-- use this to compare LuaTorch against the PyTorch equivalent

local Tensor     = require('luatorch.tensor')
local autograd   = require('luatorch.autograd')
local Linear     = require('luatorch.nn.linear')
local activation = require('luatorch.nn.activation')
local loss_fn    = require('luatorch.nn.loss')
local Sequential = require('luatorch.nn.sequential')
local Adam       = require('luatorch.optim.adam')

-- xor dataset
local inputs = Tensor.new({4, 2})
inputs:set(0, 0.0) inputs:set(1, 0.0)
inputs:set(2, 0.0) inputs:set(3, 1.0)
inputs:set(4, 1.0) inputs:set(5, 0.0)
inputs:set(6, 1.0) inputs:set(7, 1.0)

local targets = Tensor.new({4, 1})
targets:set(0, 0.0)
targets:set(1, 1.0)
targets:set(2, 1.0)
targets:set(3, 0.0)

-- build network
local model = Sequential.new(
    Linear.new(2, 16),
    activation.Tanh.new(),
    Linear.new(16, 1),
    activation.Sigmoid.new()
)

print(string.format('luatorch XOR benchmark'))
print(string.format('luajit: %s', jit and jit.version or 'unknown'))
print(string.format('model: %s', tostring(model)))
print(string.format('parameters: %d', model:num_params()))
print('')

local params    = model:parameters()
local optimizer = Adam.new(params, 0.05)
local criterion = loss_fn.MSELoss.new()

-- memory usage helper (lua heap in KB)
local function mem_kb()
    collectgarbage('collect')
    return collectgarbage('count')
end

local mem_start = mem_kb()

-- training loop with per-epoch timing
local epochs = 1000
local epoch_times = {}
local final_loss = 0

local start_time = os.clock()

for epoch = 1, epochs do
    local t0 = os.clock()

    autograd.zero_graph()
    model:zero_grad()
    autograd.watch(inputs)

    local pred = model(inputs)
    local loss = criterion(pred, targets)
    autograd.backward(loss)
    optimizer:step()

    local t1 = os.clock()
    epoch_times[epoch] = (t1 - t0) * 1000  -- ms
    final_loss = loss:get(0)
end

local total_time = os.clock() - start_time
local mem_end = mem_kb()

-- compute statistics
local function percentile(arr, p)
    local sorted = {}
    for i, v in ipairs(arr) do sorted[i] = v end
    table.sort(sorted)
    local idx = math.ceil(#sorted * p)
    if idx < 1 then idx = 1 end
    if idx > #sorted then idx = #sorted end
    return sorted[idx]
end

local sum = 0
local max_t = 0
for _, t in ipairs(epoch_times) do
    sum = sum + t
    if t > max_t then max_t = t end
end
local avg_ms = sum / epochs
local p50 = percentile(epoch_times, 0.50)
local p95 = percentile(epoch_times, 0.95)
local p99 = percentile(epoch_times, 0.99)

-- report
print('results')
print(string.rep('-', 40))
print(string.format('epochs:              %d', epochs))
print(string.format('final loss:          %.6f', final_loss))
print(string.format('total time:          %.3f sec', total_time))
print(string.format('avg time per epoch:  %.3f ms', avg_ms))
print(string.format('median per epoch:    %.3f ms', p50))
print(string.format('p95 per epoch:       %.3f ms', p95))
print(string.format('p99 per epoch:       %.3f ms', p99))
print(string.format('max per epoch:       %.3f ms', max_t))
print(string.format('epochs/sec:          %.0f', epochs / total_time))
print('')
print(string.format('lua memory start:    %.1f KB', mem_start))
print(string.format('lua memory end:      %.1f KB', mem_end))
print(string.format('lua memory delta:    %.1f KB', mem_end - mem_start))

if Tensor.cuda_memory_allocated and Tensor.cuda_memory_allocated() > 0 then
    print(string.format('gpu memory:          %.1f MB',
        Tensor.cuda_memory_allocated() / 1024 / 1024))
end

-- sanity check
print('')
print('predictions:')
autograd.enabled = false
for i = 0, 3 do
    local x1 = inputs:get(i * 2)
    local x2 = inputs:get(i * 2 + 1)
    local target = targets:get(i)
    local single = Tensor.new({1, 2})
    single:set(0, x1)
    single:set(1, x2)
    local pred = model(single)
    print(string.format('  %.0f xor %.0f = %.4f (target: %.0f)',
        x1, x2, pred:get(0), target))
end
autograd.enabled = true

-- optional: write csv
local csv = io.open('xor_benchmark.csv', 'w')
if csv then
    csv:write('epoch,time_ms\n')
    for i, t in ipairs(epoch_times) do
        csv:write(string.format('%d,%.4f\n', i, t))
    end
    csv:close()
    print('\nper-epoch timings saved to xor_benchmark.csv')
end
