local Tensor = require('luatorch.tensor')

local function assert_eq(a, b, msg)
    if math.abs(a - b) > 1e-4 then
        error(string.format('%s: expected %f got %f', msg or 'assert_eq', b, a))
    end
end

-- 2x2 matmul with known result
-- [1 2] @ [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
-- [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
local a = Tensor.new({2, 2})
a:set(0, 1); a:set(1, 2); a:set(2, 3); a:set(3, 4)

local b = Tensor.new({2, 2})
b:set(0, 5); b:set(1, 6); b:set(2, 7); b:set(3, 8)

local c = Tensor.matmul(a, b)
assert_eq(c:get(0), 19, 'matmul [0,0]')
assert_eq(c:get(1), 22, 'matmul [0,1]')
assert_eq(c:get(2), 43, 'matmul [1,0]')
assert_eq(c:get(3), 50, 'matmul [1,1]')

-- non-square matmul [2,3] x [3,2] = [2,2]
local m = Tensor.new({2, 3})
m:set(0, 1); m:set(1, 2); m:set(2, 3)
m:set(3, 4); m:set(4, 5); m:set(5, 6)

local n = Tensor.new({3, 2})
n:set(0, 7); n:set(1, 8)
n:set(2, 9); n:set(3, 10)
n:set(4, 11); n:set(5, 12)

local r = Tensor.matmul(m, n)
assert(r.shape[1] == 2 and r.shape[2] == 2, 'matmul output shape')
assert_eq(r:get(0), 1*7+2*9+3*11, 'matmul rect [0,0]')
assert_eq(r:get(1), 1*8+2*10+3*12, 'matmul rect [0,1]')

-- transpose
local t = Tensor.new({2, 3})
t:set(0, 1); t:set(1, 2); t:set(2, 3)
t:set(3, 4); t:set(4, 5); t:set(5, 6)

local tt = Tensor.transpose(t)
assert(tt.shape[1] == 3 and tt.shape[2] == 2, 'transpose shape')
assert_eq(tt:get(0), 1, 'transpose [0,0]')
assert_eq(tt:get(1), 4, 'transpose [0,1]')
assert_eq(tt:get(2), 2, 'transpose [1,0]')
assert_eq(tt:get(3), 5, 'transpose [1,1]')

-- dot product
local d1 = Tensor.new({3})
d1:set(0, 1); d1:set(1, 2); d1:set(2, 3)
local d2 = Tensor.new({3})
d2:set(0, 4); d2:set(1, 5); d2:set(2, 6)
assert_eq(Tensor.dot(d1, d2), 32, 'dot product')  -- 1*4 + 2*5 + 3*6 = 32

-- shape mismatch doesnt crash
local ok = pcall(function()
    local x = Tensor.new({2, 3})
    local y = Tensor.new({4, 2})
    Tensor.matmul(x, y)  -- returns nil
end)
