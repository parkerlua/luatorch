-- lr schedulers that wrap any optimizer
-- they adjust the learning rate during training
-- good scheduling is critical for transformer training

-- cosine annealing
-- smoothly decays lr following a cosine curve
-- starts at max_lr and ends at min_lr over T steps
local CosineAnnealing = {}
CosineAnnealing.__index = CosineAnnealing

function CosineAnnealing.new(optimizer, T_max, min_lr)
    local self = setmetatable({}, CosineAnnealing)
    self.optimizer = optimizer
    self.T_max     = T_max
    self.min_lr    = min_lr or 0.0
    self.max_lr    = optimizer.lr
    self.step_num  = 0
    return self
end

function CosineAnnealing:step()
    self.step_num = self.step_num + 1
    local t = math.min(self.step_num, self.T_max)
    local lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) *
               (1.0 + math.cos(math.pi * t / self.T_max))
    self.optimizer.lr = lr
    return lr
end

function CosineAnnealing:get_lr()
    return self.optimizer.lr
end

function CosineAnnealing:__tostring()
    return string.format('CosineAnnealing(T=%d, min=%.6f, max=%.6f, step=%d)',
        self.T_max, self.min_lr, self.max_lr, self.step_num)
end

-- warmup scheduler
-- linearly ramps lr from 0 to target over warmup_steps
-- then hands off to another scheduler (or stays constant)
local WarmupScheduler = {}
WarmupScheduler.__index = WarmupScheduler

function WarmupScheduler.new(optimizer, warmup_steps, after_scheduler)
    local self = setmetatable({}, WarmupScheduler)
    self.optimizer       = optimizer
    self.warmup_steps    = warmup_steps
    self.target_lr       = optimizer.lr
    self.after_scheduler = after_scheduler
    self.step_num        = 0
    return self
end

function WarmupScheduler:step()
    self.step_num = self.step_num + 1

    if self.step_num <= self.warmup_steps then
        -- linear warmup from 0 to target_lr
        local lr = self.target_lr * (self.step_num / self.warmup_steps)
        self.optimizer.lr = lr
        return lr
    else
        -- hand off to after_scheduler if we have one
        if self.after_scheduler then
            return self.after_scheduler:step()
        end
        return self.optimizer.lr
    end
end

function WarmupScheduler:get_lr()
    return self.optimizer.lr
end

function WarmupScheduler:__tostring()
    return string.format('WarmupScheduler(warmup=%d, step=%d)',
        self.warmup_steps, self.step_num)
end

-- step lr
-- multiply lr by gamma every step_size epochs
-- classic staircase decay
local StepLR = {}
StepLR.__index = StepLR

function StepLR.new(optimizer, step_size, gamma)
    local self = setmetatable({}, StepLR)
    self.optimizer = optimizer
    self.step_size = step_size
    self.gamma     = gamma or 0.1
    self.base_lr   = optimizer.lr
    self.epoch     = 0
    return self
end

function StepLR:step()
    self.epoch = self.epoch + 1
    local num_decays = math.floor(self.epoch / self.step_size)
    local lr = self.base_lr * (self.gamma ^ num_decays)
    self.optimizer.lr = lr
    return lr
end

function StepLR:get_lr()
    return self.optimizer.lr
end

function StepLR:__tostring()
    return string.format('StepLR(step_size=%d, gamma=%.3f, epoch=%d)',
        self.step_size, self.gamma, self.epoch)
end

return {
    CosineAnnealing  = CosineAnnealing,
    WarmupScheduler  = WarmupScheduler,
    StepLR           = StepLR,
}
