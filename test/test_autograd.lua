local Tensor   = require('luatorch.tensor')
local autograd = require('luatorch.autograd')

local function assert_eq(a, b, msg)
    if math.abs(a - b) > 1e-4 then
        error(string.format('%s: expected %f got %f', msg or 'assert_eq', b, a))
    end
end

-- gradient of add is 1 for both inputs
do
    autograd.zero_graph()
    local a = Tensor.new({2}); a:set(0, 3.0); a:set(1, 4.0)
    local b = Tensor.new({2}); b:set(0, 5.0); b:set(1, 6.0)
    autograd.watch(a)
    autograd.watch(b)

    local c = autograd.add(a, b)
    autograd.backward(c)

    assert_eq(a.grad:get(0), 1.0, 'add grad a[0]')
    assert_eq(a.grad:get(1), 1.0, 'add grad a[1]')
    assert_eq(b.grad:get(0), 1.0, 'add grad b[0]')
    assert_eq(b.grad:get(1), 1.0, 'add grad b[1]')
end

-- gradient of mul: grad_a = b, grad_b = a
do
    autograd.zero_graph()
    local a = Tensor.new({2}); a:set(0, 3.0); a:set(1, 4.0)
    local b = Tensor.new({2}); b:set(0, 5.0); b:set(1, 6.0)
    autograd.watch(a)
    autograd.watch(b)

    local c = autograd.mul(a, b)
    autograd.backward(c)

    assert_eq(a.grad:get(0), 5.0, 'mul grad a[0] = b[0]')
    assert_eq(a.grad:get(1), 6.0, 'mul grad a[1] = b[1]')
    assert_eq(b.grad:get(0), 3.0, 'mul grad b[0] = a[0]')
    assert_eq(b.grad:get(1), 4.0, 'mul grad b[1] = a[1]')
end

-- gradient of matmul
-- out = a @ b, grad_a = grad @ b^T, grad_b = a^T @ grad
do
    autograd.zero_graph()
    local a = Tensor.new({1, 2}); a:set(0, 1.0); a:set(1, 2.0)
    local b = Tensor.new({2, 1}); b:set(0, 3.0); b:set(1, 4.0)
    autograd.watch(a)
    autograd.watch(b)

    local c = autograd.matmul(a, b)  -- [1,1] = 1*3 + 2*4 = 11
    assert_eq(c:get(0), 11.0, 'matmul forward')

    autograd.backward(c)

    -- grad_a = grad @ b^T = [1] @ [3, 4] = [3, 4]
    assert_eq(a.grad:get(0), 3.0, 'matmul grad a[0]')
    assert_eq(a.grad:get(1), 4.0, 'matmul grad a[1]')

    -- grad_b = a^T @ grad = [1; 2] @ [1] = [1; 2]
    assert_eq(b.grad:get(0), 1.0, 'matmul grad b[0]')
    assert_eq(b.grad:get(1), 2.0, 'matmul grad b[1]')
end

-- gradient accumulates when tensor used twice
-- c = a + a means grad_a should be 2, not 1
do
    autograd.zero_graph()
    local a = Tensor.new({2}); a:set(0, 1.0); a:set(1, 2.0)
    autograd.watch(a)

    local c = autograd.add(a, a)
    autograd.backward(c)

    assert_eq(a.grad:get(0), 2.0, 'acc grad a+a [0]')
    assert_eq(a.grad:get(1), 2.0, 'acc grad a+a [1]')
end

-- zero_grad clears gradients
do
    local a = Tensor.new({2})
    autograd.watch(a)
    a.grad = Tensor.new({2}); a.grad:ones()
    assert(a.grad ~= nil, 'grad should exist before zero')
    autograd.zero_grad({a})
    assert(a.grad == nil, 'grad should be nil after zero_grad')
end

-- backward on 2 layer network produces nonzero gradients
do
    autograd.zero_graph()
    local x = Tensor.new({1, 2}); x:set(0, 1.0); x:set(1, 1.0)
    local w1 = Tensor.new({2, 3}); w1:randn()
    local w2 = Tensor.new({3, 1}); w2:randn()
    autograd.watch(w1)
    autograd.watch(w2)

    local h = autograd.matmul(x, w1)
    local y = autograd.matmul(h, w2)
    autograd.backward(y)

    assert(w1.grad ~= nil, 'w1 should have gradient')
    assert(w2.grad ~= nil, 'w2 should have gradient')

    -- at least one element should be nonzero
    local has_nonzero = false
    for i = 0, w1.grad:numel() - 1 do
        if math.abs(w1.grad:get(i)) > 1e-8 then has_nonzero = true break end
    end
    assert(has_nonzero, 'w1 grad should have nonzero elements')
end
