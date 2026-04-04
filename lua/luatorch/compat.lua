-- compat polyfills for LuaJIT 2.1
-- LuaJIT without LUAJIT_ENABLE_LUA52COMPAT lacks string.pack, string.unpack,
-- and table.unpack. this module polyfills them using FFI so examples and
-- checkpoint serialization work on any LuaJIT build

local ffi = require('ffi')

-- polyfill table.unpack (Lua 5.2+) as the global unpack (LuaJIT)
if not table.unpack then
    table.unpack = unpack
end

-- polyfill unpack if somehow missing
if not unpack then
    unpack = table.unpack
end

-- if string.pack/unpack already exist (compiled with 5.2 compat), skip
if string.pack and string.unpack then return end

-- format specifiers we support:
--   '>I4' = big-endian uint32
--   '>i4' = big-endian int32
--   'i4'  = little-endian int32
--   'i8'  = little-endian int64
--   'f'   = little-endian float32
--   'd'   = little-endian float64
--   'B'   = unsigned byte

local function fmt_info(fmt)
    if fmt == '>I4' or fmt == '>i4' then
        return 4, 'uint32_t', true   -- big-endian
    elseif fmt == 'i4' then
        return 4, 'int32_t', false
    elseif fmt == 'i8' then
        return 8, 'int64_t', false
    elseif fmt == 'f' then
        return 4, 'float', false
    elseif fmt == 'd' then
        return 8, 'double', false
    elseif fmt == 'B' then
        return 1, 'uint8_t', false
    else
        error('luatorch compat: unsupported pack format: ' .. tostring(fmt))
    end
end

local function swap4(s)
    return s:sub(4,4) .. s:sub(3,3) .. s:sub(2,2) .. s:sub(1,1)
end

-- string.unpack polyfill
-- signature: string.unpack(fmt, s, pos) returns value, next_pos
string.unpack = function(fmt, s, pos)
    pos = pos or 1
    local size, typ, big_endian = fmt_info(fmt)

    local chunk = s:sub(pos, pos + size - 1)
    if #chunk < size then
        error('luatorch compat: not enough bytes to unpack ' .. fmt)
    end

    -- for big-endian 4-byte ints, byte-swap before ffi copy
    if big_endian and size == 4 then
        chunk = swap4(chunk)
    end

    local buf = ffi.new(typ .. '[1]')
    ffi.copy(buf, chunk, size)

    local val
    if typ == 'int64_t' or typ == 'uint32_t' then
        val = tonumber(buf[0])
    elseif typ == 'uint8_t' then
        val = buf[0]
    else
        val = buf[0]
    end

    return val, pos + size
end

-- string.pack polyfill
-- signature: string.pack(fmt, val) returns packed string
string.pack = function(fmt, val)
    local size, typ, big_endian = fmt_info(fmt)

    local buf = ffi.new(typ .. '[1]', val)
    local s = ffi.string(buf, size)

    if big_endian and size == 4 then
        s = swap4(s)
    end

    return s
end
