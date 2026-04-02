local ffi = require('ffi')

-- distributed training support
-- detects available gpus and provides multi-gpu primitives

local dist = {}

-- try to load nccl functions from the main library
local ok, lib = pcall(ffi.load, 'luatorch')
if not ok then
    return dist
end

-- check if nccl functions are available (compiled with NCCL support)
local has_nccl = pcall(function()
    ffi.cdef[[
        int  luatorch_nccl_get_gpu_count();
        int  luatorch_nccl_init(int num_gpus);
        int  luatorch_nccl_allreduce(float** ptrs, int64_t count, int num_gpus);
        int  luatorch_nccl_broadcast(float** ptrs, int64_t count, int root, int num_gpus);
        void luatorch_nccl_destroy();
    ]]
end)

if has_nccl then
    dist.available = true
    dist.num_gpus  = tonumber(lib.luatorch_nccl_get_gpu_count())
else
    dist.available = false
    dist.num_gpus  = 0
end

function dist.init(num_gpus)
    if not dist.available then
        print('luatorch: distributed not available (nccl not found)')
        return false
    end
    num_gpus = num_gpus or dist.num_gpus
    local ret = lib.luatorch_nccl_init(num_gpus)
    return ret == 0
end

function dist.destroy()
    if dist.available then
        lib.luatorch_nccl_destroy()
    end
end

return dist
