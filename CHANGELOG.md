# Changelog

## v0.1.0 — Initial Release

### Bug Fixes (pre-release audit)

**Critical gradient accumulation bug** — All autograd backward functions used `tensor.grad = new_grad` instead of `tensor.grad += new_grad`. Tensors used multiple times (residual connections, weight sharing, `a + a`) only got the last gradient. Fixed in autograd.lua, activation.lua, loss.lua, layernorm.lua, batchnorm.lua, dropout.lua, embedding.lua, positional.lua.

**Linear bias broadcasting** — `autograd.add(out, bias)` crashed when `out` is `[batch, features]` and `bias` is `[features]` because `check_same_size` requires identical shapes. Added `tensor_add_broadcast` (CPU + CUDA) for proper `[rows, cols] + [cols]` addition with correct backward that sums gradients across the batch dimension.

**Conv2d bias dead weights** — Bias was added with raw `tensor:set()` outside autograd, so bias gradients were always zero and bias never learned. Fixed to use `add_broadcast` through autograd with proper backward.

**Flash attention crashes** — `float acc[128]` stack array crashed if `head_dim > 128`. Moved to shared memory (any size). Division by zero on `1/row_sum` when causal mask blocked all keys. Serial kernel launch per batch*head replaced with single launch.

**CUDA memory leaks** — Loss function temp tensors freed with raw `cudaFree` bypassing the memory pool. Fixed to use `pool_free`. Reduction functions had no error checks on `cudaMalloc`/`malloc`, leaked on failure. Memory pool block entry `malloc` had no NULL check.

**Label bounds checking** — Cross entropy cast `float` labels to `int64_t` without validating range. Out-of-bounds labels caused memory corruption. Fixed in CPU (loss.c) and CUDA (loss.cu, fused.cu).

**Thread safety** — cuBLAS handle creation had a race condition on simultaneous init. Fixed with `pthread_once`. Memory pool counter reads had no mutex. Fixed.

**NCCL broadcast API** — Took single `float*` but GPUs have separate address spaces. Changed to `float**` (per-GPU pointers).

**Tensor operations on GPU tensors** — `tensor_get`, `tensor_set`, `tensor_print`, `tensor_copy` crashed when `t->data` was NULL (GPU tensors). Fixed to check and warn.

**Box-Muller randn** — `rand_float()` could return 0.0, causing `logf(0) = -inf`. Fixed with `rand_float_safe()`.

**Adam truncated** — adam.lua file was cut off at `-- cha`. Completed with `set_lr()` and `__tostring()`.

**Softmax single-threaded** — CUDA softmax used `<<<rows, 1>>>` (one thread per row). Parallelized with 256 threads per row using shared memory reduction.

**Fused layernorm single-threaded** — Same issue, same fix.

**Broadcast backward serial** — One thread per column looping all rows. Parallelized with `atomicAdd`, one thread per element.

**Checkpoint dtype** — `string.pack('f')` for everything silently corrupted non-float32 data. Fixed to write dtype code and use correct pack format per type.

**Config tostring** — `#self.values` always returned 0 for hash tables. Fixed to iterate with `pairs()`.

**ONNX double encoding** — Protobuf dimensions were double-wrapped in length-delimited fields. Fixed encoding.

**Flash attention seq_len** — Relied on undocumented `query._seq_len` field that nothing set. Changed to explicit parameter.

**NCCL Lua wrappers** — `allreduce` and `broadcast` accessed `cuda_data` without checking tensor device. Added guards.

### Performance Optimizations

- **LuaJIT FFI caching**: 80+ hot-path C function calls cached as local variables for better trace compilation
- **Async CUDA transfers**: `cuda_async()`/`cpu_async()` using `cudaMemcpyAsync` with dedicated stream
- **Autograd graph pruning**: intermediate gradients freed after backward, ~2x less peak gradient memory
- **C im2col**: conv2d im2col moved from Lua nested loops to C, ~100x faster
- **Zero-copy reshape**: multihead attention 3D-to-2D uses `reshape()` instead of element-by-element copy
- **LayerNorm caching**: forward caches `inv_std` and `x_hat`, backward reuses them instead of recomputing 3x
- **Fused Adam on CUDA**: optimizers auto-dispatch to single-kernel fused adam when tensors are on GPU


### Core
- Tensor type with shape, strides, dtype, device, ref counting, gradient pointer
- CPU implementations: fill, zeros, ones, rand, randn (Box-Muller), copy, reshape, get, set, print
- Elementwise ops: add, sub, mul, div, scalar variants, neg, abs, sqrt, log, exp
- Reductions: sum, mean, max, min
- Inplace ops: add_, sub_, mul_scalar_, add_scalar_
- Matrix ops: matmul, batched matmul, transpose, dot product
- Activations: relu, sigmoid, tanh, gelu, silu, softmax
- Loss functions: MSE, MAE, cross entropy (forward + backward)

### CUDA
- Device transfer: tensor_to_cuda, tensor_to_cpu
- CUDA kernels for every CPU operation with auto-dispatch
- cuBLAS matmul with tensor core support (cublasSgemm, cublasSgemmStridedBatched)
- Parallel reduction kernels with shared memory for sum, max, min
- Flash Attention 2 with tiled computation and causal masking
- Fused kernels: LayerNorm, Adam, GELU+bias, cross entropy
- FP16 cast kernels (float32 to/from float16)
- CUDA memory pool with block reuse and thread safety

### Autograd
- Computation graph with Node tracking
- Forward/backward for add, mul, matmul, relu
- Gradient accumulation (+=) for tensors used multiple times
- zero_graph, zero_grad, watch, backward

### Neural Network Layers
- Linear (kaiming init, optional bias)
- Conv2d (im2col implementation)
- Embedding (sparse backward)
- Dropout (train/eval modes, inverted scaling)
- LayerNorm (learnable gamma/beta, analytical backward)
- BatchNorm1d (running stats, train/eval modes)
- MultiHeadAttention (Q/K/V projections, scaled dot-product)
- FlashMultiHeadAttention (uses flash attention kernel on CUDA)
- TransformerBlock (pre-norm, residual connections, 4x FFN)
- SinusoidalPE, LearnedPE (positional encodings)
- ReLU, Sigmoid, Tanh, GELU, SiLU (activation modules)
- MSELoss, MAELoss, CrossEntropyLoss
- Sequential (layer container)

### Optimizers
- Adam (momentum, velocity, bias correction)
- AdamW (decoupled weight decay, no-decay parameter lists)
- SGD (momentum, optional Nesterov)
- CosineAnnealing, WarmupScheduler, StepLR schedulers

### Data
- DataLoader (batching, Fisher-Yates shuffle)
- Character Tokenizer (encode/decode, save/load vocab)
- TextDataset (text file loading, train/val split, batch generation)

### Models
- GPT (embedding + positional encoding + N transformer blocks + output head, text generation with temperature and top-k sampling)
- MLP (auto-built from layer size list)

### I/O
- Binary checkpoint save/load with shape verification
- ONNX export with pure-Lua protobuf writer

### Mixed Precision
- GradScaler with loss scaling, overflow detection, dynamic scale adjustment

### Distributed
- NCCL wrapper (allreduce, broadcast)
- DistributedDataParallel (gradient averaging across GPUs)

### Infrastructure
- One-command installer with GPU auto-detection
- Comprehensive benchmark suite
- Training logger with CSV export
- Config file loader with defaults
- Clean `nn.Layer()` callable syntax
- Master entry point: `require('luatorch')` exports everything
- Full test suite (tensor, arithmetic, matmul, autograd, nn, optim, training, cuda)
- GitHub Actions CI pipeline
- README with complete API reference
- Contributing guide with step-by-step examples

### Examples
- XOR network
- MNIST digit classification
- CIFAR-10 image classification
- GPT language model on Shakespeare (AMP + distributed + ONNX export)
- PyTorch comparison document
