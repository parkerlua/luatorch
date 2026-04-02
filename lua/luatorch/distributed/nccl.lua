local ffi    = require('ffi')
local Tensor = require('luatorch.tensor')

-- thin lua wrapper around nccl operations
-- used by DDP to synchronize gradients across gpus

local lib = ffi.load('luatorch')

local nccl = {}

-- average a list of tensors across all gpus
-- tensors must all be on different gpus and have the same shape
function nccl.allreduce(tensors)
    local num_gpus = #tensors
    local count    = tensors[1]:numel()

    -- build array of cuda data pointers
    local ptrs = ffi.new('float*[?]', num_gpus)
    for i, t in ipairs(tensors) do
        ptrs[i - 1] = t._raw.cuda_data
    end

    local ret = lib.luatorch_nccl_allreduce(ptrs, count, num_gpus)
    if ret ~= 0 then
        error('luatorch error: nccl allreduce failed')
    end
end

-- broadcast tensor from root gpu to all others
function nccl.broadcast(tensor, root, num_gpus)
    root = root or 0
    local ret = lib.luatorch_nccl_broadcast(
        tensor._raw.cuda_data, tensor:numel(), root, num_gpus)
    if ret ~= 0 then
        error('luatorch error: nccl broadcast failed')
    end
end

return nccl
