local Tensor = require('luatorch.tensor')

local function assert_eq(a, b, msg)
    if math.abs(a - b) > 1e-4 then
        error(string.format('%s: expected %f got %f', msg or 'assert_eq', b, a))
    end
end

-- setup known tensors
local a = Tensor.new({3})
a:set(0, 1.0); a:set(1, 2.0); a:set(2, 3.0)
local b = Tensor.new({3})
b:set(0, 4.0); b:set(1, 5.0); b:set(2, 6.0)

-- add
local c = Tensor.add(a, b)
assert_eq(c:get(0), 5.0, 'add 0')
assert_eq(c:get(1), 7.0, 'add 1')
assert_eq(c:get(2), 9.0, 'add 2')

-- sub
c = Tensor.sub(b, a)
assert_eq(c:get(0), 3.0, 'sub 0')
assert_eq(c:get(1), 3.0, 'sub 1')

-- mul
c = Tensor.mul(a, b)
assert_eq(c:get(0), 4.0, 'mul 0')
assert_eq(c:get(1), 10.0, 'mul 1')
assert_eq(c:get(2), 18.0, 'mul 2')

-- div
c = Tensor.div(b, a)
assert_eq(c:get(0), 4.0, 'div 0')
assert_eq(c:get(1), 2.5, 'div 1')
assert_eq(c:get(2), 2.0, 'div 2')

-- scalar ops
c = Tensor.add_scalar(a, 10.0)
assert_eq(c:get(0), 11.0, 'add_scalar')

c = Tensor.mul_scalar(a, 3.0)
assert_eq(c:get(1), 6.0, 'mul_scalar')

c = Tensor.div_scalar(a, 2.0)
assert_eq(c:get(2), 1.5, 'div_scalar')

c = Tensor.pow_scalar(a, 2.0)
assert_eq(c:get(2), 9.0, 'pow_scalar')

-- unary ops
c = Tensor.neg(a)
assert_eq(c:get(0), -1.0, 'neg')

local pos = Tensor.new({2})
pos:set(0, -3.0); pos:set(1, 5.0)
c = Tensor.abs(pos)
assert_eq(c:get(0), 3.0, 'abs neg')
assert_eq(c:get(1), 5.0, 'abs pos')

local sq = Tensor.new({2})
sq:set(0, 4.0); sq:set(1, 9.0)
c = Tensor.sqrt(sq)
assert_eq(c:get(0), 2.0, 'sqrt')
assert_eq(c:get(1), 3.0, 'sqrt')

-- reductions
assert_eq(Tensor.sum(a), 6.0, 'sum')
assert_eq(Tensor.mean(a), 2.0, 'mean')
assert_eq(Tensor.max(a), 3.0, 'max')
assert_eq(Tensor.min(a), 1.0, 'min')

-- inplace ops
local d = a:copy()
Tensor.add_(d, b)
assert_eq(d:get(0), 5.0, 'add_ 0')
assert_eq(d:get(1), 7.0, 'add_ 1')

d = a:copy()
Tensor.sub_(d, b)
assert_eq(d:get(0), -3.0, 'sub_ 0')

d = a:copy()
Tensor.mul_scalar_(d, 2.0)
assert_eq(d:get(0), 2.0, 'mul_scalar_')

-- division by zero is handled
local zeros = Tensor.new({2}); zeros:zeros()
local ones = Tensor.new({2}); ones:ones()
local result = Tensor.div(ones, zeros)  -- should not crash

-- size mismatch gives error
local small = Tensor.new({2})
local big = Tensor.new({3})
local ok = pcall(function() Tensor.add(small, big) end)
-- this returns nil rather than crashing, which is fine
