local Tensor = require('luatorch.tensor')

-- checkpoint saves and loads model parameters to disk
-- uses a simple binary format:
-- for each parameter tensor:
--   4 bytes: ndim
--   ndim * 8 bytes: shape (int64 each)
--   numel * 4 bytes: float32 data

local checkpoint = {}

-- save all parameters from a model to a file
-- model must have a parameters() method
function checkpoint.save(model, path)
    local params = model:parameters()
    local f = io.open(path, 'wb')
    if not f then
        error('luatorch error: could not open file for writing: ' .. path)
    end

    -- write number of parameters
    local n_params = #params
    f:write(string.pack('i4', n_params))

    for _, param in ipairs(params) do
        -- if on gpu, we need to read from cpu
        -- the get function handles this through the C layer
        local ndim  = param.ndim
        local shape = param.shape
        local numel = param:numel()

        -- write ndim
        f:write(string.pack('i4', ndim))

        -- write shape
        for i = 1, ndim do
            f:write(string.pack('i8', shape[i]))
        end

        -- write data
        for i = 0, numel - 1 do
            f:write(string.pack('f', param:get(i)))
        end
    end

    f:close()
end

-- load parameters from a file into a model
-- model must have a parameters() method
-- shapes must match what was saved
function checkpoint.load(model, path)
    local params = model:parameters()
    local f = io.open(path, 'rb')
    if not f then
        error('luatorch error: could not open file for reading: ' .. path)
    end

    -- read number of parameters
    local n_params = string.unpack('i4', f:read(4))

    if n_params ~= #params then
        f:close()
        error(string.format(
            'luatorch error: parameter count mismatch, file has %d but model has %d',
            n_params, #params))
    end

    for idx, param in ipairs(params) do
        -- read ndim
        local ndim = string.unpack('i4', f:read(4))

        -- read shape
        local shape = {}
        for i = 1, ndim do
            shape[i] = string.unpack('i8', f:read(8))
        end

        -- verify shape matches
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

        -- read data
        local numel = param:numel()
        for i = 0, numel - 1 do
            local val = string.unpack('f', f:read(4))
            param:set(i, val)
        end
    end

    f:close()
end

return checkpoint
