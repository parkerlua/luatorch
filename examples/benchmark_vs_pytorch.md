# luatorch vs PyTorch — Side by Side

## Building a Transformer

### luatorch

```lua
local lf = require('luatorch')
local nn = lf.nn

local model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
)

local optimizer = lf.optim.AdamW.new(model:parameters(), 3e-4, 0.9, 0.999, 1e-8, 0.01)
local criterion = nn.CrossEntropyLoss()
```

### PyTorch

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
```

## Training Loop

### luatorch

```lua
for epoch = 1, 10 do
    for batch_data, batch_targets in train_loader:iter() do
        lf.autograd.zero_graph()
        model:zero_grad()
        lf.autograd.watch(batch_data)

        local pred = model(batch_data)
        local loss = criterion(pred, batch_targets)

        lf.autograd.backward(loss)
        optimizer:step()
    end
end
```

### PyTorch

```python
for epoch in range(10):
    for batch_data, batch_targets in train_loader:
        optimizer.zero_grad()

        pred = model(batch_data)
        loss = criterion(pred, batch_targets)

        loss.backward()
        optimizer.step()
```

## GPT Model

### luatorch

```lua
local gpt = lf.models.GPT.new(vocab_size, 384, 6, 6, 256, 0.1)
local tokens = gpt:generate(prompt, 200, 0.8, 40)
```

### PyTorch

```python
# requires writing the full GPT class yourself or using HuggingFace
from transformers import GPT2LMHeadModel, GPT2Config
config = GPT2Config(vocab_size=vocab_size, n_embd=384, n_head=6, n_layer=6)
gpt = GPT2LMHeadModel(config)
```

## Performance on RTX 4090

These are representative numbers for a 6-layer, 6-head, 384-dim transformer with sequence length 128:

| Metric | luatorch | PyTorch | Notes |
|--------|---------|---------|-------|
| Framework startup | ~10ms | ~2000ms | LuaJIT vs Python import |
| Matmul 4096x4096 | Both use cuBLAS | Both use cuBLAS | Same performance, both use tensor cores |
| Flash Attention | Custom kernel | torch.nn.functional.scaled_dot_product_attention | PyTorch's is more optimized |
| Memory overhead | ~50MB | ~500MB | LuaJIT has a tiny runtime |
| Training tokens/sec | Competitive | Benchmark | cuBLAS dominates both |

### Where luatorch wins
- **Startup time**: LuaJIT loads in milliseconds, Python takes seconds
- **Memory footprint**: LuaJIT runtime is ~2MB vs Python's ~50MB+
- **Simplicity**: 51 files total, you can read the entire framework
- **Embedding**: LuaJIT embeds anywhere (games, embedded systems, edge devices)
- **Hackability**: Every operation is one C function + one Lua wrapper, no abstraction layers

### Where PyTorch wins
- **Flash Attention quality**: PyTorch has FlashAttention-2 from the original authors
- **Ecosystem**: Thousands of pretrained models, HuggingFace, torchvision
- **Distributed**: Battle-tested FSDP, DeepSpeed integration
- **Autograd**: Dynamic graph with full backward for every op, not just the core ops
- **Community**: Millions of users, extensive documentation

## Memory Usage

| Model | luatorch | PyTorch |
|-------|---------|---------|
| MLP (784->256->10) | ~2MB | ~50MB |
| GPT (6L, 384d) | ~100MB | ~400MB |
| GPT + AMP | ~80MB | ~300MB |

The difference is mostly Python runtime overhead. GPU memory usage for model weights and activations is similar since both use cuBLAS.

## When to Use luatorch

- You know Lua and want to do AI without learning Python
- You need to embed AI in a Lua application (games, scripting engines)
- You want to understand how a deep learning framework works end to end
- You want minimal dependencies and a small footprint
- You are doing research and want to modify the framework internals easily

## When to Use PyTorch

- You need pretrained models and the HuggingFace ecosystem
- You need production distributed training at scale
- You need the widest possible operator coverage
- You need automatic differentiation for custom operations beyond the core set
- Your team already knows Python
