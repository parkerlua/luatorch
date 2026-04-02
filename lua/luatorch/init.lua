-- luatorch — modern deep learning framework for Lua
-- require('luatorch') gives you everything

local lf = {}

lf.version = '0.1.0'

lf.Tensor   = require('luatorch.tensor')
lf.autograd = require('luatorch.autograd')
lf.nn       = require('luatorch.nn')

lf.optim = {
    Adam             = require('luatorch.optim.adam'),
    AdamW            = require('luatorch.optim.adamw'),
    SGD              = require('luatorch.optim.sgd'),
    CosineAnnealing  = require('luatorch.optim.scheduler').CosineAnnealing,
    WarmupScheduler  = require('luatorch.optim.scheduler').WarmupScheduler,
    StepLR           = require('luatorch.optim.scheduler').StepLR,
}

lf.data = {
    DataLoader  = require('luatorch.data.dataloader'),
    Tokenizer   = require('luatorch.data.tokenizer'),
    TextDataset = require('luatorch.data.dataset'),
}

lf.models = {
    GPT = require('luatorch.models.gpt'),
    MLP = require('luatorch.models.mlp'),
}

lf.io = {
    checkpoint = require('luatorch.io.checkpoint'),
}

lf.utils = {
    Logger = require('luatorch.utils.logger'),
    Config = require('luatorch.utils.config'),
}

-- amp (optional, needs cuda)
local ok_amp, amp = pcall(require, 'luatorch.cuda.amp')
if ok_amp then
    lf.cuda = { amp = amp }
else
    lf.cuda = {}
end

-- distributed (optional, needs nccl)
local ok_dist, dist = pcall(require, 'luatorch.distributed')
if ok_dist and dist.available then
    lf.distributed = {
        DDP      = require('luatorch.distributed.ddp'),
        nccl     = require('luatorch.distributed.nccl'),
        num_gpus = dist.num_gpus,
        init     = dist.init,
        destroy  = dist.destroy,
    }
else
    lf.distributed = nil
end

-- onnx export
local ok_onnx, onnx_mod = pcall(require, 'luatorch.export.onnx')
if ok_onnx then
    lf.export = { onnx = onnx_mod }
else
    lf.export = {}
end

-- print version info in debug mode
if os.getenv('LUAFLOW_DEBUG') then
    print(string.format('luatorch v%s', lf.version))
    if lf.distributed then
        print(string.format('  gpus: %d', lf.distributed.num_gpus))
    end
end

return lf
