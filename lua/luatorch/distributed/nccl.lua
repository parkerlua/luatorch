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

    local ptrs = ffi.new('float*[?]', num_gpus)
    for i, t in ipairs(tensors) do
        -- fix: check tensor is on cuda before accessing cuda_data
        if t.device ~= 'cuda' then
            error('luatorch error: nccl allreduce requires all tensors on cuda')
        end
        ptrs[i - 1] = t._raw.cuda_data
    end

    local ret = lib.luatorch_nccl_allreduce(ptrs, count, num_gpus)
    if ret ~= 0 then
        error('luatorch error: nccl allreduce failed')
    end
end

-- fix: broadcast now takes a list of tensors (per-GPU pointers)
-- old API took a single tensor pointer which is wrong because GPUs have separate address spaces
function nccl.broadcast(tensors, root, num_gpus)
    root = root or 0
    num_gpus = num_gpus or #tensors
    local count = tensors[1]:numel()

    local ptrs = ffi.new('float*[?]', num_gpus)
    for i, t in ipairs(tensors) do
        if t.device ~= 'cuda' then
            error('luatorch error: nccl broadcast requires all tensors on cuda')
        end
        ptrs[i - 1] = t._raw.cuda_data
    end

    local ret = lib.luatorch_nccl_broadcast(ptrs, count, root, num_gpus)
    if ret ~= 0 then
        error('luatorch error: nccl broadcast failed')
    end
end

return nccl
