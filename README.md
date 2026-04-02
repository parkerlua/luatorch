# luatorch

A modern CUDA-accelerated deep learning framework written in Lua and C. The first serious Lua AI framework since Torch7 was abandoned in 2018.

luatorch gives you the full PyTorch experience in Lua: tensors, autograd, neural network layers, optimizers, data loading, model checkpointing, and GPU acceleration. Everything from training an XOR network to training a GPT language model on your 4090.

## Why Lua?

LuaJIT is the fastest dynamic language runtime ever built. Its FFI lets you call C functions with zero overhead. Lua has a tiny footprint, embeds anywhere, and the language is simple enough that you can read the entire framework source in an afternoon. If you know Lua and have a GPU, you can do real AI research without Python.

## Hardware Requirements

- **LuaJIT** 2.1+ (not standard Lua — LuaJIT is required for FFI)
- **CUDA Toolkit** 12.0+ (for GPU acceleration)
- **CMake** 3.18+
- **Any NVIDIA GPU** with compute capability 5.0+ (tested on RTX 4090)
- **Linux or macOS** for development (training runs on Linux with NVIDIA drivers)

## Installation

```bash
git clone https://github.com/youruser/luatorch.git
cd luatorch
bash tools/install.sh
```

The installer checks all dependencies, detects your GPU architecture, builds the C/CUDA library, and runs a smoke test. If anything is missing it tells you exactly what to install.

Manual build:

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
make -j$(nproc)
cp libluatorch.so ../lua/luatorch/
cd ..
luajit examples/xor.lua
```

## Quick Start

```lua
local lf = require('luatorch')
local nn = lf.nn

-- build a model
local model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
)

-- train it
local optimizer = lf.optim.Adam.new(model:parameters(), 0.001)
local criterion = nn.CrossEntropyLoss()

for epoch = 1, 10 do
    lf.autograd.zero_graph()
    model:zero_grad()

    local pred = model(input_batch)
    local loss = criterion(pred, targets)

    lf.autograd.backward(loss)
    optimizer:step()

    print(string.format('epoch %d  loss: %.4f', epoch, loss:get(0)))
end

-- save it
lf.io.checkpoint.save(model, 'model.bin')
```

## API Reference

### Tensor

```lua
local Tensor = lf.Tensor

-- creation
local t = Tensor.new({3, 4})              -- 3x4 zeros
local t = Tensor.new({3, 4}, 'float32', 'cpu')

-- fill
t:zeros()  t:ones()  t:rand()  t:randn()  t:fill(3.14)

-- element access
t:get(0)           -- 0-indexed flat access
t:set(0, 1.0)

-- device transfer
t:cuda()           -- move to GPU
t:cpu()            -- move back

-- arithmetic (auto-dispatches CPU/CUDA)
Tensor.add(a, b)   Tensor.sub(a, b)   Tensor.mul(a, b)   Tensor.div(a, b)
Tensor.add_scalar(a, 2.0)   Tensor.mul_scalar(a, 0.5)
Tensor.neg(a)  Tensor.abs(a)  Tensor.sqrt(a)  Tensor.log(a)  Tensor.exp(a)

-- operators
local c = a + b    -- uses __add metamethod
local c = a * b
local c = -a

-- reductions (return Lua numbers)
Tensor.sum(a)  Tensor.mean(a)  Tensor.max(a)  Tensor.min(a)

-- matmul (cuBLAS on GPU with tensor core support)
Tensor.matmul(a, b)    -- [M,K] x [K,N] -> [M,N]
Tensor.bmm(a, b)       -- batched matmul
Tensor.transpose(a)
Tensor.dot(a, b)

-- activations
Tensor.relu(a)  Tensor.sigmoid(a)  Tensor.tanh(a)  Tensor.gelu(a)  Tensor.silu(a)
Tensor.softmax(a)

-- loss
Tensor.mse_loss(pred, target)
Tensor.cross_entropy_loss(pred, target)

-- fused ops (CUDA only, falls back to separate ops on CPU)
Tensor.fused_gelu_bias(x, bias)
Tensor.fused_layernorm(x, gamma, beta, eps)
Tensor.fused_adam(param, grad, m, v, lr, beta1, beta2, eps, bc1, bc2, wd)
Tensor.flash_attention(Q, K, V, batch_heads, seq_len, head_dim, causal)

-- memory pool
Tensor.pool_init()
Tensor.pool_clear()
Tensor.cuda_memory_allocated()
Tensor.cuda_memory_cached()
```

### Autograd

```lua
local autograd = lf.autograd

autograd.watch(tensor)         -- track gradients for this tensor
autograd.zero_graph()          -- clear computation graph
autograd.zero_grad(params)     -- zero all parameter gradients

-- tracked operations (build graph automatically)
local out = autograd.add(a, b)
local out = autograd.mul(a, b)
local out = autograd.matmul(a, b)
local out = autograd.relu(x)

-- backward pass
autograd.backward(loss)        -- compute all gradients

autograd.enabled = false       -- disable tracking for inference
```

### Neural Network Layers (nn)

All layers support the clean callable syntax: `nn.Linear(784, 256)` instead of `Linear.new(784, 256)`.

```lua
local nn = lf.nn

-- layers
nn.Linear(in_features, out_features, bias)     -- default bias=true
nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
nn.Embedding(vocab_size, embed_dim)
nn.Dropout(p)                                  -- default p=0.5
nn.LayerNorm(dim, eps)
nn.BatchNorm1d(features, eps, momentum)

-- containers
nn.Sequential(layer1, layer2, ...)

-- attention
nn.MultiHeadAttention(embed_dim, num_heads)
nn.FlashMultiHeadAttention(embed_dim, num_heads, causal)  -- uses flash attention on CUDA
nn.TransformerBlock(embed_dim, num_heads, dropout, ffn_mult, use_flash)

-- positional encoding
nn.SinusoidalPE(max_seq_len, embed_dim)
nn.LearnedPE(max_seq_len, embed_dim)

-- activations (as layers for Sequential)
nn.ReLU()  nn.Sigmoid()  nn.Tanh()  nn.GELU()  nn.SiLU()

-- loss functions
nn.MSELoss()  nn.MAELoss()  nn.CrossEntropyLoss()

-- all layers implement:
layer:forward(input)           -- or layer(input)
layer:parameters()             -- returns list of trainable tensors
layer:zero_grad()
layer:num_params()
tostring(layer)                -- prints layer info
```

### Optimizers

```lua
local optim = lf.optim

optim.Adam.new(params, lr, beta1, beta2, eps)
optim.AdamW.new(params, lr, beta1, beta2, eps, weight_decay, no_decay_keys)
optim.SGD.new(params, lr, momentum, nesterov)

optimizer:step()               -- update parameters
optimizer:reset()
optimizer:set_lr(new_lr)
```

### Learning Rate Schedulers

```lua
optim.CosineAnnealing.new(optimizer, T_max, min_lr)
optim.WarmupScheduler.new(optimizer, warmup_steps, after_scheduler)
optim.StepLR.new(optimizer, step_size, gamma)

scheduler:step()               -- returns new lr
scheduler:get_lr()
```

### Data Loading

```lua
local data = lf.data

-- batch loader with shuffle
local loader = data.DataLoader.new(data_tensor, target_tensor, batch_size, shuffle)
for batch_data, batch_targets in loader:iter() do
    -- train on batch
end

-- character tokenizer
local tokenizer = data.Tokenizer.new(text)
local ids = tokenizer:encode("hello")
local text = tokenizer:decode(ids)
tokenizer:save("vocab.txt")
local tok = data.Tokenizer.load("vocab.txt")

-- text dataset for language modeling
local dataset = data.TextDataset.new("data.txt", seq_len, "train", 0.9)
local input, target = dataset:get_batch(batch_size)
```

### Pre-built Models

```lua
local models = lf.models

-- GPT language model
local gpt = models.GPT.new(vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout)
local logits = gpt(input_tokens)
local generated = gpt:generate(prompt, max_tokens, temperature, top_k)

-- configurable MLP
local mlp = models.MLP.new({784, 256, 128, 10})
```

### I/O

```lua
-- checkpoint save/load
lf.io.checkpoint.save(model, "model.bin")
lf.io.checkpoint.load(model, "model.bin")

-- ONNX export
lf.export.onnx.export(model, dummy_input, "model.onnx")
```

### Utilities

```lua
-- training logger
local logger = lf.utils.Logger.new(print_every)
logger:log(loss, accuracy, lr)
logger:save_csv("training.csv")

-- config loader
local config = lf.utils.Config.new(defaults)
config:load("config.lua")
config:get("lr")
config:print()
```

### Automatic Mixed Precision

```lua
local scaler = lf.cuda.amp.GradScaler.new(init_scale, growth_factor, backoff_factor, growth_interval)

-- training loop
local scaled_loss = scaler:scale_loss(loss)
autograd.backward(scaled_loss)
scaler:step(optimizer)     -- unscales grads, skips step on overflow
scaler:update()            -- adjusts scale factor
```

### Distributed Training (Multi-GPU)

```lua
local dist = lf.distributed

-- wrap model for data parallel training
local ddp_model = dist.DDP.new(model)

-- training loop uses ddp_model instead of model
-- gradients are automatically averaged across GPUs via NCCL
```

## Performance

### Flash Attention
The flash attention implementation avoids materializing the N x N attention matrix. For sequence length 2048 with 384-dim embeddings, this saves ~16MB of GPU memory per attention layer and runs 2-3x faster than naive attention.

### Fused CUDA Kernels
- **Fused Adam**: single kernel launch instead of 8+ separate operations per parameter update
- **Fused LayerNorm**: mean, variance, normalize, scale, shift in one pass
- **Fused GELU+Bias**: activation and bias add combined
- **Fused CrossEntropy**: softmax + NLL in one pass

### CUDA Memory Pool
Reuses GPU memory allocations instead of calling cudaMalloc/cudaFree on every tensor operation. Reduces allocation overhead by 10-50x depending on workload.

### cuBLAS
Matrix multiplication uses cuBLAS `cublasSgemm` which automatically uses tensor cores on Ampere/Ada GPUs (RTX 3090, 4090). No manual tensor core programming needed.

## Architecture

### C/Lua FFI Bridge
All tensor operations are implemented in C (CPU) and CUDA (GPU). LuaJIT's FFI calls these functions with zero overhead — no Lua C API marshalling. The Lua `Tensor` type wraps a C `Tensor*` pointer and auto-dispatches to CPU or CUDA implementations based on `tensor.device`.

### Autograd
The computation graph is a simple list of `Node` objects. Each node records the operation, inputs, output tensor, and a gradient function. `autograd.backward(loss)` walks the list in reverse and calls each gradient function. Gradients accumulate with `+=` so tensors used multiple times (residual connections, weight sharing) get correct gradients.

### CUDA Dispatch
Every Tensor operation checks `tensor.device` and calls the corresponding `_cuda` function if on GPU. This happens transparently — the same Lua code works on CPU and GPU without changes.

```lua
-- this code works on both CPU and CUDA
local c = Tensor.add(a, b)  -- calls tensor_add or tensor_add_cuda automatically
```

## Examples

- `examples/xor.lua` — XOR network, the hello world of neural nets
- `examples/mnist.lua` — MNIST digit classification, downloads data automatically
- `examples/gpt.lua` — GPT language model on Shakespeare, with AMP and distributed support
- `examples/cifar.lua` — CIFAR-10 image classification

## Running Benchmarks

```bash
luajit tools/benchmark.lua
```

Benchmarks every major operation at multiple sizes, reports GFLOPS and tokens/sec, compares CPU vs CUDA, saves results to CSV.

## Contributing

1. Fork and create a feature branch
2. Keep the same style — simple comments, no fancy dividers, no unnecessary abstractions
3. Every C function needs a matching Lua wrapper
4. Every CUDA kernel needs `cudaGetLastError` error checking
5. Test with `luajit examples/xor.lua` at minimum before submitting
6. New layers need `parameters()`, `zero_grad()`, `num_params()`, `__tostring()`, and `__call`

## License

MIT License. Use it however you want.
