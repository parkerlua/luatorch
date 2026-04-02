-- config loader
-- loads hyperparameters from a lua config file
-- merges with defaults so you only need to specify what you change
-- prints full config at start of training

local Config = {}
Config.__index = Config

-- defaults = table of default hyperparameters
function Config.new(defaults)
    local self = setmetatable({}, Config)
    self.values = {}

    -- copy defaults
    if defaults then
        for k, v in pairs(defaults) do
            self.values[k] = v
        end
    end

    return self
end

-- load config from a lua file
-- the file should return a table of values
-- example config file:
--   return {
--       lr = 0.001,
--       batch_size = 64,
--       epochs = 10,
--   }
function Config:load(path)
    local f = io.open(path, 'r')
    if not f then
        print('config: using defaults, no config file at ' .. path)
        return self
    end
    f:close()

    local chunk, err = loadfile(path)
    if not chunk then
        error('config: failed to load ' .. path .. ': ' .. err)
    end

    local overrides = chunk()
    if type(overrides) ~= 'table' then
        error('config: file must return a table')
    end

    -- merge overrides into current values
    for k, v in pairs(overrides) do
        self.values[k] = v
    end

    return self
end

-- get a config value
function Config:get(key, default)
    local val = self.values[key]
    if val == nil then return default end
    return val
end

-- set a config value
function Config:set(key, value)
    self.values[key] = value
end

-- print full config in a readable format
function Config:print()
    print('config:')
    -- sort keys for consistent output
    local keys = {}
    for k, _ in pairs(self.values) do
        table.insert(keys, k)
    end
    table.sort(keys)

    for _, k in ipairs(keys) do
        local v = self.values[k]
        print(string.format('  %-20s = %s', k, tostring(v)))
    end
end

function Config:__tostring()
    -- fix: count hash table entries properly, # only works on sequences
    local count = 0
    for _ in pairs(self.values) do count = count + 1 end
    return string.format('Config(%d values)', count)
end

return Config
