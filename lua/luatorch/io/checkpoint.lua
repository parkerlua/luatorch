local Tensor = require('luatorch.tensor')

-- checkpoint saves and loads model parameters to disk
-- binary format per parameter:
--   4 bytes: ndim
--   4 bytes: dtype code (0=float32, 1=float64, 2=int32, 3=int64)
--   ndim * 8 bytes: shape (int64 each)
--   numel * dtype_size bytes: data

local checkpoint = {}

-- fix: pack format and byte size per dtype
-- old code used 'f' (float32) for everything, silently corrupting float64/int data
local dtype_pack = {
    float32 = {fmt = 'f', size = 4, code = 0},
    float64 = {fmt = 'd', size = 8, code = 1},
    int32   = {fmt = 'i4', size = 4, code = 2},
    int64   = {fmt = 'i8', size = 8, code = 3},
}

local code_to_dtype = {}
for name, info in pairs(dtype_pack) do
    code_to_dtype[info.code] = name
end

function checkpoint.save(model, path)
    local params = model:parameters()
    local f = io.open(path, 'wb')
    if not f then
        error('luatorch error: could not open file for writing: ' .. path)
    end

    local n_params = #params
    f:write(string.pack('i4', n_params))

    for _, param in ipairs(params) do
        local ndim  = param.ndim
        local shape = param.shape
        local numel = param:numel()
        local dtype = param.dtype or 'float32'
        local info  = dtype_pack[dtype] or dtype_pack.float32

        f:write(string.pack('i4', ndim))
        f:write(string.pack('i4', info.code))

        for i = 1, ndim do
            f:write(string.pack('i8', shape[i]))
        end

        for i = 0, numel - 1 do
            f:write(string.pack(info.fmt, param:get(i)))
        end
    end

    f:close()
end

function checkpoint.load(model, path)
    local params = model:parameters()
    local f = io.open(path, 'rb')
    if not f then
        error('luatorch error: could not open file for reading: ' .. path)
    end

    local n_params = string.unpack('i4', f:read(4))

    if n_params ~= #params then
        f:close()
        error(string.format(
            'luatorch error: parameter count mismatch, file has %d but model has %d',
            n_params, #params))
    end

    for idx, param in ipairs(params) do
        local ndim = string.unpack('i4', f:read(4))

        -- read dtype code, default to float32 for backwards compatibility with old checkpoints
        local dtype_code_raw = f:read(4)
        local dtype_code = 0
        if dtype_code_raw then
            dtype_code = string.unpack('i4', dtype_code_raw)
        end
        local dtype_name = code_to_dtype[dtype_code] or 'float32'
        local info = dtype_pack[dtype_name]

        local shape = {}
        for i = 1, ndim do
            shape[i] = string.unpack('i8', f:read(8))
        end

        if ndim ~= param.ndim then
            f:close()
            error(string.format(
                'luatorch error: ndim mismatch for parameter %d, file has %d but model has %d',
                idx, ndim, param.ndim))
        end

        for i = 1, ndim do
            if shape[i] ~= param.shape[i] then
                f:close()
                error(string.format(
                    'luatorch error: shape mismatch for parameter %d dimension %d',
                    idx, i))
            end
        end

        local numel = param:numel()
        for i = 0, numel - 1 do
            local val = string.unpack(info.fmt, f:read(info.size))
            param:set(i, val)
        end
    end

    f:close()
end

return checkpoint
