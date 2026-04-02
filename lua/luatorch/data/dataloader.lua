local Tensor = require('luatorch.tensor')

-- dataloader takes a dataset and serves it in batches
-- shuffles the data each epoch so the model sees different orderings
-- this is important because seeing the same order every time
-- can cause the model to learn spurious patterns

local DataLoader = {}
DataLoader.__index = DataLoader

-- data     = tensor of inputs, first dimension is number of samples
-- targets  = tensor of labels/targets, first dimension matches data
-- batch_size = how many samples per batch
-- shuffle  = whether to shuffle each epoch, default true
function DataLoader.new(data, targets, batch_size, shuffle)
    local self = setmetatable({}, DataLoader)

    self.data       = data
    self.targets    = targets
    self.batch_size = batch_size or 32
    self.shuffle    = shuffle ~= false
    self.n_samples  = data.shape[1]

    -- build index array for shuffling
    self.indices = {}
    for i = 1, self.n_samples do
        self.indices[i] = i
    end

    return self
end

-- fisher-yates shuffle
local function shuffle_indices(indices)
    for i = #indices, 2, -1 do
        local j = math.random(1, i)
        indices[i], indices[j] = indices[j], indices[i]
    end
end

-- figure out the shape of one sample
-- if data is [N, 784] then sample shape is {784}
-- if data is [N, 3, 32, 32] then sample shape is {3, 32, 32}
local function sample_shape(tensor)
    local shape = {}
    for i = 2, tensor.ndim do
        table.insert(shape, tensor.shape[i])
    end
    return shape
end

-- how many elements per sample (everything after first dim)
local function sample_size(tensor)
    local n = 1
    for i = 2, tensor.ndim do
        n = n * tensor.shape[i]
    end
    return n
end

-- return number of batches per epoch
function DataLoader:num_batches()
    return math.ceil(self.n_samples / self.batch_size)
end

-- iterate over batches
-- returns an iterator function that yields (batch_data, batch_targets) tensors
function DataLoader:iter()
    if self.shuffle then
        shuffle_indices(self.indices)
    end

    local data_size   = sample_size(self.data)
    local target_size = sample_size(self.targets)
    local s_shape     = sample_shape(self.data)
    local t_shape     = sample_shape(self.targets)
    local pos         = 1
    local total       = self.n_samples
    local bs          = self.batch_size
    local indices     = self.indices

    return function()
        if pos > total then return nil end

        -- figure out actual batch size (last batch may be smaller)
        local actual_bs = math.min(bs, total - pos + 1)

        -- build batch data shape: {actual_bs, ...sample_shape}
        local batch_d_shape = {actual_bs}
        for _, s in ipairs(s_shape) do table.insert(batch_d_shape, s) end

        local batch_t_shape = {actual_bs}
        for _, s in ipairs(t_shape) do table.insert(batch_t_shape, s) end

        local batch_data    = Tensor.new(batch_d_shape)
        local batch_targets = Tensor.new(batch_t_shape)

        -- copy samples into batch tensors
        for b = 0, actual_bs - 1 do
            local idx = indices[pos + b] - 1  -- convert to 0-indexed

            -- copy data sample
            for j = 0, data_size - 1 do
                batch_data:set(b * data_size + j,
                              self.data:get(idx * data_size + j))
            end

            -- copy target sample
            for j = 0, target_size - 1 do
                batch_targets:set(b * target_size + j,
                                 self.targets:get(idx * target_size + j))
            end
        end

        pos = pos + actual_bs
        return batch_data, batch_targets
    end
end

function DataLoader:__tostring()
    return string.format('DataLoader(samples=%d, batch_size=%d, batches=%d)',
        self.n_samples, self.batch_size, self:num_batches())
end

return DataLoader
