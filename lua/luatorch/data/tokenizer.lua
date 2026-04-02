local Tensor = require('luatorch.tensor')

-- character level tokenizer
-- maps individual characters to integer ids and back
-- simple but effective for small scale language modeling
-- build vocab from any text, save/load for reuse

local Tokenizer = {}
Tokenizer.__index = Tokenizer

-- build a tokenizer from a text string
function Tokenizer.new(text)
    local self = setmetatable({}, Tokenizer)

    -- build character set from the text
    local chars = {}
    local seen  = {}
    for i = 1, #text do
        local c = text:sub(i, i)
        if not seen[c] then
            seen[c] = true
            table.insert(chars, c)
        end
    end

    -- sort for deterministic ordering
    table.sort(chars)

    -- build lookup tables
    self.char_to_id = {}
    self.id_to_char = {}
    self.vocab_size = #chars

    for i, c in ipairs(chars) do
        local id = i - 1  -- 0-indexed
        self.char_to_id[c] = id
        self.id_to_char[id] = c
    end

    return self
end

-- encode a string into a list of integer ids
function Tokenizer:encode(text)
    local ids = {}
    for i = 1, #text do
        local c = text:sub(i, i)
        local id = self.char_to_id[c]
        if id == nil then
            error('luatorch error: unknown character: ' .. c)
        end
        table.insert(ids, id)
    end
    return ids
end

-- encode into a tensor
function Tokenizer:encode_tensor(text)
    local ids = self:encode(text)
    local t = Tensor.new({#ids})
    for i, id in ipairs(ids) do
        t:set(i - 1, id)
    end
    return t
end

-- decode a list of integer ids back to string
function Tokenizer:decode(ids)
    local chars = {}
    for _, id in ipairs(ids) do
        local c = self.id_to_char[id]
        if c == nil then
            error('luatorch error: unknown token id: ' .. id)
        end
        table.insert(chars, c)
    end
    return table.concat(chars)
end

-- decode a tensor back to string
function Tokenizer:decode_tensor(tensor)
    local ids = {}
    for i = 0, tensor:numel() - 1 do
        table.insert(ids, math.floor(tensor:get(i)))
    end
    return self:decode(ids)
end

-- save vocab to a file
-- simple format: one character per line (with its id)
function Tokenizer:save(path)
    local f = io.open(path, 'w')
    if not f then error('could not open: ' .. path) end

    f:write(string.format('%d\n', self.vocab_size))
    for id = 0, self.vocab_size - 1 do
        local c = self.id_to_char[id]
        -- store as byte value to handle whitespace characters
        f:write(string.format('%d %d\n', id, string.byte(c)))
    end

    f:close()
end

-- load vocab from a file
function Tokenizer.load(path)
    local f = io.open(path, 'r')
    if not f then error('could not open: ' .. path) end

    local self = setmetatable({}, Tokenizer)
    self.char_to_id = {}
    self.id_to_char = {}

    self.vocab_size = tonumber(f:read('*l'))

    for _ = 1, self.vocab_size do
        local line = f:read('*l')
        local id, byte_val = line:match('(%d+) (%d+)')
        id = tonumber(id)
        byte_val = tonumber(byte_val)
        local c = string.char(byte_val)
        self.char_to_id[c] = id
        self.id_to_char[id] = c
    end

    f:close()
    return self
end

function Tokenizer:__tostring()
    return string.format('Tokenizer(vocab_size=%d)', self.vocab_size)
end

return Tokenizer
