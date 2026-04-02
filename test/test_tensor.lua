local Tensor = require('luatorch.tensor')

local function assert_eq(a, b, msg)
    if math.abs(a - b) > 1e-5 then
        error(string.format('%s: expected %f got %f', msg or 'assert_eq', b, a))
    end
end

-- creation
local t = Tensor.new({3, 4})
assert(t.ndim == 2, 'ndim should be 2')
assert(t.shape[1] == 3, 'shape[1] should be 3')
assert(t.shape[2] == 4, 'shape[2] should be 4')
assert(t:numel() == 12, 'numel should be 12')

-- different shapes
local t1d = Tensor.new({5})
assert(t1d.ndim == 1 and t1d:numel() == 5, '1d creation')

local t3d = Tensor.new({2, 3, 4})
assert(t3d.ndim == 3 and t3d:numel() == 24, '3d creation')

-- fill ops
t:zeros()
assert_eq(t:get(0), 0, 'zeros')

t:ones()
assert_eq(t:get(0), 1, 'ones')
assert_eq(t:get(11), 1, 'ones last')

t:fill(3.14)
assert_eq(t:get(5), 3.14, 'fill')

-- rand and randn produce values
t:rand()
local has_nonzero = false
for i = 0, t:numel() - 1 do
    if t:get(i) ~= 0 then has_nonzero = true break end
end
assert(has_nonzero, 'rand should produce nonzero values')

t:randn()
has_nonzero = false
for i = 0, t:numel() - 1 do
    if t:get(i) ~= 0 then has_nonzero = true break end
end
assert(has_nonzero, 'randn should produce nonzero values')

-- get and set
local s = Tensor.new({3})
s:set(0, 1.5)
s:set(1, 2.5)
s:set(2, 3.5)
assert_eq(s:get(0), 1.5, 'set/get 0')
assert_eq(s:get(1), 2.5, 'set/get 1')
assert_eq(s:get(2), 3.5, 'set/get 2')

-- copy is independent
local a = Tensor.new({3})
a:set(0, 10.0); a:set(1, 20.0); a:set(2, 30.0)
local b = a:copy()
b:set(0, 99.0)
assert_eq(a:get(0), 10.0, 'copy should be independent')
assert_eq(b:get(0), 99.0, 'copy should have new value')

-- tostring doesnt crash
local str = tostring(a)
assert(str:find('Tensor'), 'tostring should contain Tensor')

-- print doesnt crash
a:print()
