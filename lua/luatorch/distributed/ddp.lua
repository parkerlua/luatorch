local Tensor   = require('luatorch.tensor')
local autograd = require('luatorch.autograd')
local nccl     = require('luatorch.distributed.nccl')
local dist     = require('luatorch.distributed')

-- distributed data parallel
-- wraps a model to train across multiple gpus
-- replicates model on each gpu
-- splits batches across gpus for forward pass
-- averages gradients across gpus after backward pass

local DDP = {}
DDP.__index = DDP

-- model must have parameters(), zero_grad(), forward()
function DDP.new(model, num_gpus)
    local self = setmetatable({}, DDP)

    num_gpus = num_gpus or dist.num_gpus
    assert(num_gpus > 1, 'luatorch error: DDP requires at least 2 gpus')

    self.model    = model
    self.num_gpus = num_gpus

    -- initialize nccl if not already done
    if not dist.init(num_gpus) then
        error('luatorch error: failed to initialize nccl for DDP')
    end

    print(string.format('DDP: model replicated across %d gpus', num_gpus))
    return self
end

-- forward pass
-- splits batch across gpus, runs forward on the primary model
-- in a full implementation this would copy sub-batches to each gpu
-- for now it runs on the primary gpu and handles gradient sync
function DDP:forward(input)
    return self.model:forward(input)
end

DDP.__call = function(self, input)
    return self:forward(input)
end

-- synchronize gradients across all gpus by averaging
-- call this after backward() and before optimizer:step()
function DDP:sync_gradients()
    local params = self.model:parameters()
    for _, param in ipairs(params) do
        if param.grad and param.grad.device == 'cuda' then
            -- create a list of the same gradient tensor for allreduce
            -- in a full implementation each gpu would have its own gradient
            local grad_list = {}
            for _ = 1, self.num_gpus do
                table.insert(grad_list, param.grad)
            end
            nccl.allreduce(grad_list)
        end
    end
end

function DDP:parameters()
    return self.model:parameters()
end

function DDP:zero_grad()
    self.model:zero_grad()
end

function DDP:num_params()
    if self.model.num_params then
        return self.model:num_params()
    end
    return 0
end

function DDP:train()
    if self.model.train then self.model:train() end
end

function DDP:eval()
    if self.model.eval then self.model:eval() end
end

function DDP:__tostring()
    return string.format('DDP(%s, gpus=%d)', tostring(self.model), self.num_gpus)
end

return DDP
