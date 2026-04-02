-- cuda tests
-- skip gracefully if cuda is not available

local Tensor = require('luatorch.tensor')

-- check if cuda is available by trying to init the pool
local cuda_available = pcall(function()
    Tensor.pool_init()
    -- try to allocate a tiny tensor on gpu
    local t = Tensor.new({2})
    t:cuda()
    t:cpu()
end)

if not cuda_available then
    error('SKIP: cuda not available')
end

local autograd = require('luatorch.autograd')

local function assert_near(a, b, tol, msg)
    tol = tol or 1e-4
    if math.abs(a - b) > tol then
        error(string.format('%s: expected ~%f got %f', msg or 'assert_near', b, a))
    end
end

-- tensor moves to gpu and back correctly
do
    local t = Tensor.new({4})
    t:set(0, 1.0); t:set(1, 2.0); t:set(2, 3.0); t:set(3, 4.0)

    t:cuda()
    assert(t.device == 'cuda', 'should be on cuda')

    t:cpu()
    assert(t.device == 'cpu', 'should be back on cpu')

    assert_near(t:get(0), 1.0, 1e-5, 'round trip [0]')
    assert_near(t:get(1), 2.0, 1e-5, 'round trip [1]')
    assert_near(t:get(2), 3.0, 1e-5, 'round trip [2]')
    assert_near(t:get(3), 4.0, 1e-5, 'round trip [3]')
end

-- gpu add produces same result as cpu
do
    local a = Tensor.new({4}); a:set(0, 1); a:set(1, 2); a:set(2, 3); a:set(3, 4)
    local b = Tensor.new({4}); b:set(0, 5); b:set(1, 6); b:set(2, 7); b:set(3, 8)

    -- cpu result
    local cpu_result = Tensor.add(a, b)

    -- gpu result
    a:cuda(); b:cuda()
    local gpu_result = Tensor.add(a, b)
    gpu_result:cpu()

    for i = 0, 3 do
        assert_near(cpu_result:get(i), gpu_result:get(i), 1e-5,
            string.format('gpu add mismatch at %d', i))
    end
end

-- gpu matmul produces same result as cpu
do
    local a = Tensor.new({2, 2}); a:set(0, 1); a:set(1, 2); a:set(2, 3); a:set(3, 4)
    local b = Tensor.new({2, 2}); b:set(0, 5); b:set(1, 6); b:set(2, 7); b:set(3, 8)

    local cpu_result = Tensor.matmul(a, b)

    a:cuda(); b:cuda()
    local gpu_result = Tensor.matmul(a, b)
    gpu_result:cpu()

    for i = 0, 3 do
        assert_near(cpu_result:get(i), gpu_result:get(i), 1e-3,
            string.format('gpu matmul mismatch at %d', i))
    end
end

-- memory pool reports nonzero usage after allocations
do
    local allocated = Tensor.cuda_memory_allocated()
    assert(allocated > 0, 'should have allocated some gpu memory')
end

print('  cuda tests passed')
