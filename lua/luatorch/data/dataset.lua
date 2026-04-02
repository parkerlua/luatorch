local Tensor    = require('luatorch.tensor')
local Tokenizer = require('luatorch.data.tokenizer')

-- text dataset for language modeling
-- loads a text file, tokenizes it, returns overlapping windows
-- each sample is (input, target) where target is input shifted by one

local TextDataset = {}
TextDataset.__index = TextDataset

-- path      = path to text file
-- seq_len   = sequence length for each training window
-- split     = 'train' or 'val', splits at 90% by default
-- split_pct = where to split, default 0.9
function TextDataset.new(path, seq_len, split, split_pct)
    local self = setmetatable({}, TextDataset)

    split     = split     or 'train'
    split_pct = split_pct or 0.9

    -- read the text file
    local f = io.open(path, 'r')
    if not f then error('could not open: ' .. path) end
    local text = f:read('*a')
    f:close()

    -- build tokenizer from full text
    self.tokenizer = Tokenizer.new(text)
    self.vocab_size = self.tokenizer.vocab_size
    self.seq_len = seq_len

    -- tokenize
    local all_ids = self.tokenizer:encode(text)

    -- split into train and val
    local split_idx = math.floor(#all_ids * split_pct)
    local ids
    if split == 'train' then
        ids = {}
        for i = 1, split_idx do ids[i] = all_ids[i] end
    else
        ids = {}
        for i = split_idx + 1, #all_ids do
            table.insert(ids, all_ids[i])
        end
    end

    -- store as tensor
    self.data = Tensor.new({#ids})
    for i, id in ipairs(ids) do
        self.data:set(i - 1, id)
    end

    -- number of valid samples
    -- each sample needs seq_len + 1 tokens (input + 1 shifted target)
    self.n_samples = #ids - seq_len

    print(string.format('TextDataset(%s): %d tokens, %d samples, vocab=%d',
        split, #ids, self.n_samples, self.vocab_size))

    return self
end

-- get a single sample at index (0-based)
-- returns input [seq_len] and target [seq_len]
function TextDataset:get(idx)
    local input  = Tensor.new({self.seq_len})
    local target = Tensor.new({self.seq_len})

    for i = 0, self.seq_len - 1 do
        input:set(i,  self.data:get(idx + i))
        target:set(i, self.data:get(idx + i + 1))
    end

    return input, target
end

-- get a random batch
-- returns input [batch, seq_len] and target [batch, seq_len]
function TextDataset:get_batch(batch_size)
    local input  = Tensor.new({batch_size, self.seq_len})
    local target = Tensor.new({batch_size, self.seq_len})

    for b = 0, batch_size - 1 do
        local idx = math.random(0, self.n_samples - 1)
        for i = 0, self.seq_len - 1 do
            input:set(b * self.seq_len + i,  self.data:get(idx + i))
            target:set(b * self.seq_len + i, self.data:get(idx + i + 1))
        end
    end

    return input, target
end

function TextDataset:__tostring()
    return string.format('TextDataset(samples=%d, seq_len=%d, vocab=%d)',
        self.n_samples, self.seq_len, self.vocab_size)
end

return TextDataset
