# Changelog

## v0.1.0 — Initial Release

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
