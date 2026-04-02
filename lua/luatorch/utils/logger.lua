-- training logger
-- tracks loss, accuracy, learning rate over training
-- prints formatted tables and saves to csv

local Logger = {}
Logger.__index = Logger

-- print_every = print a summary every N steps
function Logger.new(print_every)
    local self = setmetatable({}, Logger)

    self.print_every = print_every or 100
    self.step        = 0
    self.epoch       = 0

    -- history for csv export
    self.history = {}

    -- running averages for current print window
    self.running_loss = 0.0
    self.running_acc  = 0.0
    self.running_n    = 0
    self.current_lr   = 0.0

    return self
end

-- log a training step
function Logger:log(loss, accuracy, lr)
    self.step = self.step + 1
    self.running_loss = self.running_loss + (loss or 0)
    self.running_acc  = self.running_acc  + (accuracy or 0)
    self.running_n    = self.running_n + 1
    self.current_lr   = lr or self.current_lr

    -- store in history
    table.insert(self.history, {
        step     = self.step,
        epoch    = self.epoch,
        loss     = loss or 0,
        accuracy = accuracy or 0,
        lr       = self.current_lr,
    })

    -- print if needed
    if self.step % self.print_every == 0 then
        self:print_summary()
    end
end

-- set current epoch
function Logger:set_epoch(epoch)
    self.epoch = epoch
end

-- print formatted summary
function Logger:print_summary()
    if self.running_n == 0 then return end

    local avg_loss = self.running_loss / self.running_n
    local avg_acc  = self.running_acc  / self.running_n

    local parts = {string.format('step %6d', self.step)}

    if self.epoch > 0 then
        table.insert(parts, string.format('epoch %3d', self.epoch))
    end

    table.insert(parts, string.format('loss %.4f', avg_loss))

    if avg_acc > 0 then
        table.insert(parts, string.format('acc %.2f%%', avg_acc * 100))
    end

    if self.current_lr > 0 then
        table.insert(parts, string.format('lr %.2e', self.current_lr))
    end

    print(table.concat(parts, '  '))

    -- reset running averages
    self.running_loss = 0.0
    self.running_acc  = 0.0
    self.running_n    = 0
end

-- save training history to csv
function Logger:save_csv(path)
    local f = io.open(path, 'w')
    if not f then error('could not open: ' .. path) end

    f:write('step,epoch,loss,accuracy,lr\n')
    for _, entry in ipairs(self.history) do
        f:write(string.format('%d,%d,%.6f,%.6f,%.8f\n',
            entry.step, entry.epoch, entry.loss, entry.accuracy, entry.lr))
    end

    f:close()
end

function Logger:__tostring()
    return string.format('Logger(step=%d, history=%d entries)',
        self.step, #self.history)
end

return Logger
