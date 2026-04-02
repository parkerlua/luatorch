# Contributing to LuaTorch

## Building from Source

```bash
# install dependencies
sudo apt install luajit cmake nvidia-cuda-toolkit  # ubuntu/debian

# build
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
make -j$(nproc)
cp libluatorch.so ../lua/luatorch/
cd ..
```

Or use the installer:
```bash
bash tools/install.sh
```

## Running Tests

```bash
LUA_PATH='lua/?.lua;lua/?/init.lua;;' luajit test/run_all.lua
```

CUDA tests are skipped automatically if no GPU is available. All other tests run on CPU.

## Adding a New Layer

Here's how to add a new layer, using `Tanh` as an example:

1. Create the Lua file at `lua/luatorch/nn/tanh.lua`:

```lua
local Tensor   = require('luatorch.tensor')
local autograd = require('luatorch.autograd')

local Tanh = {}
Tanh.__index = Tanh

function Tanh.new()
    return setmetatable({}, Tanh)
end

function Tanh:forward(input)
    local out = Tensor.tanh(input)

    autograd.record("tanh", {input}, out, function(grad)
        if input.requires_grad then
            local sq = Tensor.mul(out, out)
            local ones = Tensor.new(out.shape)
            ones:ones()
            local dtanh = Tensor.sub(ones, sq)
            autograd.acc_grad(input, Tensor.mul(grad, dtanh))
        end
    end)

    return out
end

Tanh.__call = function(self, input)
    return self:forward(input)
end

function Tanh:parameters() return {} end
function Tanh:zero_grad() end
function Tanh:num_params() return 0 end
function Tanh:__tostring() return 'Tanh()' end

return Tanh
```

Key requirements for every layer:
- `forward(input)` — the computation
- `__call` — so the layer is callable like a function
- `parameters()` — returns list of trainable tensors (empty for activations)
- `zero_grad()` — clears gradients
- `num_params()` — total trainable parameter count
- `__tostring()` — readable description
- Use `autograd.acc_grad()` not `input.grad =` in backward functions
- Register backward with `autograd.record()`

2. Add it to `lua/luatorch/nn/init.lua`:

```lua
Tanh = callable(require('luatorch.nn.tanh')),
```

3. Add a test in `test/test_nn.lua`.

## Adding a New CUDA Kernel

Example: adding a custom `clamp` kernel.

1. Create `csrc/cuda/ops/clamp.cu`:

```c
#include "../../tensor.h"
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" Tensor* tensor_new_cuda(int64_t* shape, int ndim, DType dtype);

#define BLOCK_SIZE 256

__global__ void clamp_kernel(float* a, float* out, float min_val, float max_val, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = a[i];
        if (val < min_val) val = min_val;
        if (val > max_val) val = max_val;
        out[i] = val;
    }
}

extern "C" Tensor* tensor_clamp_cuda(Tensor* a, float min_val, float max_val) {
    if (!a) return NULL;
    Tensor* out = tensor_new_cuda(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    int blocks = (int)((a->numel + BLOCK_SIZE - 1) / BLOCK_SIZE);
    clamp_kernel<<<blocks, BLOCK_SIZE>>>(a->cuda_data, out->cuda_data, min_val, max_val, a->numel);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "luatorch cuda kernel error: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    return out;
}
```

Key requirements:
- Use 256 threads per block
- Check `cudaGetLastError()` after every kernel launch
- Use `extern "C"` for all functions called from Lua
- Include NULL checks on inputs

2. Add the `.cu` file to `CUDA_SOURCES` in `CMakeLists.txt`.

3. Add the FFI declaration in `lua/luatorch/tensor.lua`:

```lua
-- in the ffi.cdef block
Tensor* tensor_clamp_cuda(Tensor* a, float min_val, float max_val);
```

4. Add the Lua wrapper:

```lua
function Tensor.clamp(a, min_val, max_val)
    if is_cuda(a) then
        return wrap_result(lib.tensor_clamp_cuda(a._raw, min_val, max_val), a)
    end
    -- cpu fallback
    local out = Tensor.new(a.shape)
    for i = 0, a:numel() - 1 do
        local val = a:get(i)
        if val < min_val then val = min_val end
        if val > max_val then val = max_val end
        out:set(i, val)
    end
    return out
end
```

## Code Style

- **Comments**: Simple and brief. No divider lines (`-----` or `=====`). No decorative banners.
- **C**: C99 standard. `snake_case` for functions. Prefix all tensor functions with `tensor_`.
- **Lua**: LuaJIT compatible. No Lua 5.3+ features. Metatables for OOP.
- **CUDA**: 256 threads per block default. Always check `cudaGetLastError()`. Use `extern "C"` for FFI-callable functions.
- **Error messages**: Start with `luatorch error:` for C errors. Use Lua `error()` for Lua errors.
- **No over-engineering**: Don't add abstraction layers, feature flags, or backwards compatibility shims. If you're not sure if something is needed, it isn't.

## Pull Requests

1. Fork the repo and create a branch from `main`
2. Keep changes focused — one feature or fix per PR
3. Run `luajit test/run_all.lua` and make sure all tests pass
4. Add tests for new functionality
5. Update README if adding user-facing features
6. Describe what you changed and why in the PR description
