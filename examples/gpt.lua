-- tiny GPT language model
-- trains on shakespeare and generates text
-- supports FP16 mixed precision, multi-GPU, and ONNX export

local lf = require('luatorch')

local Tensor      = lf.Tensor
local autograd    = lf.autograd
local GPT         = lf.models.GPT
local TextDataset = lf.data.TextDataset
local AdamW       = lf.optim.AdamW
local Logger      = lf.utils.Logger
local Config      = lf.utils.Config
local checkpoint  = lf.io.checkpoint
local scheduler   = require('luatorch.optim.scheduler')

-- config
local config = Config.new({
    -- model
    embed_dim    = 384,
    num_heads    = 6,
    num_layers   = 6,
    max_seq_len  = 256,
    dropout      = 0.1,
    -- training
    batch_size   = 4,
    seq_len      = 128,
    lr           = 3e-4,
    weight_decay = 0.01,
    warmup_steps = 100,
    max_steps    = 5000,
    -- features
    use_amp      = false,  -- set to true for fp16 training on gpu
    use_ddp      = false,  -- set to true for multi-gpu training
    export_onnx  = false,  -- set to true to export onnx at end
    -- logging
    print_every  = 100,
    sample_every = 500,
    save_every   = 1000,
    -- generation
    temperature  = 0.8,
    top_k        = 40,
    gen_tokens   = 200,
})

config:load('gpt_config.lua')
config:print()

-- download shakespeare if not present
local data_path = 'data/shakespeare.txt'
os.execute('mkdir -p data')
local f = io.open(data_path, 'r')
if f then
    f:close()
else
    print('downloading shakespeare...')
    os.execute('curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o ' .. data_path)
end

-- load dataset
local seq_len = config:get('seq_len')
local train_data = TextDataset.new(data_path, seq_len, 'train')
local val_data   = TextDataset.new(data_path, seq_len, 'val')

local vocab_size = train_data.vocab_size
local tokenizer  = train_data.tokenizer

-- build model
local model = GPT.new(
    vocab_size,
    config:get('embed_dim'),
    config:get('num_heads'),
    config:get('num_layers'),
    config:get('max_seq_len'),
    config:get('dropout')
)

print(tostring(model))

-- initialize memory pool
Tensor.pool_init()

-- distributed: wrap model in DDP if multiple gpus available
local ddp_model = nil
if config:get('use_ddp') and lf.distributed then
    local num_gpus = lf.distributed.num_gpus
    if num_gpus > 1 then
        local DDP = lf.distributed.DDP
        ddp_model = DDP.new(model, num_gpus)
        print(string.format('distributed training enabled: %d gpus', num_gpus))
    else
        print('only 1 gpu available, running single-gpu')
    end
end

local train_model = ddp_model or model

-- setup optimizer with weight decay
local params   = train_model:parameters()
local no_decay = model:no_decay_params()
local optimizer = AdamW.new(params, config:get('lr'),
    0.9, 0.999, 1e-8, config:get('weight_decay'), no_decay)

-- lr schedule: warmup then cosine decay
local max_steps     = config:get('max_steps')
local warmup_steps  = config:get('warmup_steps')
local cosine        = scheduler.CosineAnnealing.new(optimizer, max_steps - warmup_steps, 1e-5)
local lr_scheduler  = scheduler.WarmupScheduler.new(optimizer, warmup_steps, cosine)

-- loss
local criterion = lf.nn.CrossEntropyLoss()

-- amp scaler
local scaler = nil
local use_amp = config:get('use_amp')
if use_amp and lf.cuda and lf.cuda.amp then
    scaler = lf.cuda.amp.GradScaler.new()
    print('AMP enabled, initial scale: ' .. scaler:get_scale())
end

-- logger
local logger = Logger.new(config:get('print_every'))

-- generate sample text
local function generate_sample(prompt_text, n_tokens)
    local prompt = tokenizer:encode_tensor(prompt_text)
    local prompt_2d = Tensor.new({1, prompt:numel()})
    for i = 0, prompt:numel() - 1 do
        prompt_2d:set(i, prompt:get(i))
    end

    local output = model:generate(prompt_2d, n_tokens,
        config:get('temperature'), config:get('top_k'))
    return tokenizer:decode_tensor(output)
end

-- training loop
local batch_size    = config:get('batch_size')
local print_every   = config:get('print_every')
local sample_every  = config:get('sample_every')
local save_every    = config:get('save_every')
local gen_tokens    = config:get('gen_tokens')

print(string.format('\ntraining for %d steps...', max_steps))

local start_time = os.clock()
local tokens_processed = 0

for step = 1, max_steps do
    -- get a random batch
    local input, target = train_data:get_batch(batch_size)

    -- clear state
    autograd.zero_graph()
    train_model:zero_grad()

    -- forward pass
    autograd.watch(input)
    local logits = train_model(input)

    -- reshape target for cross entropy
    local flat_target = Tensor.new({batch_size * seq_len})
    for i = 0, batch_size * seq_len - 1 do
        flat_target:set(i, target:get(i))
    end

    local loss = criterion(logits, flat_target)

    -- backward pass
    if scaler then
        local scaled_loss = scaler:scale_loss(loss)
        autograd.backward(scaled_loss)
        if ddp_model then ddp_model:sync_gradients() end
        scaler:step(optimizer)
        scaler:update()
    else
        autograd.backward(loss)
        if ddp_model then ddp_model:sync_gradients() end
        optimizer:step()
    end

    -- update lr
    local lr = lr_scheduler:step()

    -- track throughput
    tokens_processed = tokens_processed + batch_size * seq_len

    -- log
    local loss_val = loss:get(0)
    logger:log(loss_val, nil, lr)

    -- print tokens/sec periodically
    if step % print_every == 0 then
        local elapsed = os.clock() - start_time
        local tok_per_sec = tokens_processed / elapsed
        print(string.format('  tokens/sec: %.0f', tok_per_sec))
        if scaler then
            print(string.format('  amp scale: %.0f', scaler:get_scale()))
        end
        if Tensor.cuda_memory_allocated() > 0 then
            print(string.format('  gpu mem: %.1f MB allocated, %.1f MB cached',
                Tensor.cuda_memory_allocated() / 1024 / 1024,
                Tensor.cuda_memory_cached() / 1024 / 1024))
        end
    end

    -- generate sample text periodically
    if step % sample_every == 0 then
        print('\n--- sample ---')
        local text = generate_sample('\n', gen_tokens)
        print(text)
        print('--- end sample ---\n')
    end

    -- save checkpoint periodically
    if step % save_every == 0 then
        local path = string.format('gpt_step_%d.bin', step)
        checkpoint.save(model, path)
        print('saved checkpoint: ' .. path)
    end
end

-- save final model
checkpoint.save(model, 'gpt_final.bin')
print('\nsaved final model to gpt_final.bin')

-- export to onnx if requested
if config:get('export_onnx') and lf.export and lf.export.onnx then
    local dummy_input = Tensor.new({1, config:get('seq_len')})
    for i = 0, config:get('seq_len') - 1 do dummy_input:set(i, 0) end
    lf.export.onnx.export(model, dummy_input, 'gpt_model.onnx')
    lf.export.onnx.verify('gpt_model.onnx')
end

-- save training log
logger:save_csv('gpt_training.csv')
print('saved training log to gpt_training.csv')

-- final throughput
local total_time = os.clock() - start_time
print(string.format('\ntotal time: %.1f sec', total_time))
print(string.format('avg tokens/sec: %.0f', tokens_processed / total_time))

-- cleanup distributed
if lf.distributed then lf.distributed.destroy() end

-- final generation
print('\n--- final generation ---')
print(generate_sample('ROMEO:', 300))
