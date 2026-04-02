local Tensor = require('luatorch.tensor')

-- automatic mixed precision training
-- keeps master weights in float32 for accuracy
-- runs forward pass computations faster on gpu
-- uses loss scaling to prevent gradient underflow in fp16

-- GradScaler handles the loss scaling logic
local GradScaler = {}
GradScaler.__index = GradScaler

-- init_scale      = initial loss scale factor, default 65536
-- growth_factor   = multiply scale by this when no overflow, default 2
-- backoff_factor  = multiply scale by this when overflow detected, default 0.5
-- growth_interval = grow scale every N successful steps, default 2000
function GradScaler.new(init_scale, growth_factor, backoff_factor, growth_interval)
    local self = setmetatable({}, GradScaler)

    self.scale           = init_scale      or 65536.0
    self.growth_factor   = growth_factor   or 2.0
    self.backoff_factor  = backoff_factor  or 0.5
    self.growth_interval = growth_interval or 2000
    self.growth_tracker  = 0
    self.found_inf       = false

    return self
end

-- scale the loss before backward pass
-- returns scaled_loss = loss * scale_factor
function GradScaler:scale_loss(loss)
    local scale = self.scale
    -- multiply loss value by scale
    local scaled = Tensor.mul_scalar(loss, scale)
    -- store scale for unscaling gradients later
    self._current_scale = scale
    return scaled
end

-- unscale gradients and call optimizer step
-- skips the step if gradients contain inf/nan
function GradScaler:step(optimizer)
    local inv_scale = 1.0 / self._current_scale
    self.found_inf = false

    -- unscale all gradients and check for inf/nan
    for _, param in ipairs(optimizer.params) do
        if param.grad then
            Tensor.scale_(param.grad, inv_scale)
            if Tensor.has_inf_nan(param.grad) then
                self.found_inf = true
            end
        end
    end

    -- only step if gradients are clean
    if not self.found_inf then
        optimizer:step()
    end
end

-- update the scale factor based on whether gradients overflowed
-- call after step()
function GradScaler:update()
    if self.found_inf then
        -- overflow detected, reduce scale
        self.scale = self.scale * self.backoff_factor
        self.growth_tracker = 0
    else
        -- no overflow, increment tracker
        self.growth_tracker = self.growth_tracker + 1
        if self.growth_tracker >= self.growth_interval then
            -- enough successful steps, increase scale
            self.scale = self.scale * self.growth_factor
            self.growth_tracker = 0
        end
    end
end

function GradScaler:get_scale()
    return self.scale
end

function GradScaler:__tostring()
    return string.format('GradScaler(scale=%.1f, growth=%d/%d)',
        self.scale, self.growth_tracker, self.growth_interval)
end

return {
    GradScaler = GradScaler,
}
