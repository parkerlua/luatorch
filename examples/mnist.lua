-- mnist digit classification
-- downloads the dataset, trains a simple 3 layer network
-- this is the hello world of deep learning

local Tensor     = require('luatorch.tensor')
local autograd   = require('luatorch.autograd')
local Linear     = require('luatorch.nn.linear')
local Dropout    = require('luatorch.nn.dropout')
local activation = require('luatorch.nn.activation')
local loss_fn    = require('luatorch.nn.loss')
local Sequential = require('luatorch.nn.sequential')
local Adam       = require('luatorch.optim.adam')
local DataLoader = require('luatorch.data.dataloader')
local checkpoint = require('luatorch.io.checkpoint')

-- mnist binary format reader
-- the files have a simple header then raw bytes
local function read_mnist_images(path)
    local f = io.open(path, 'rb')
    if not f then error('could not open: ' .. path) end

    -- header: magic(4), count(4), rows(4), cols(4)
    local header = f:read(16)
    local magic  = string.unpack('>I4', header, 1)
    local count  = string.unpack('>I4', header, 5)
    local rows   = string.unpack('>I4', header, 9)
    local cols   = string.unpack('>I4', header, 13)

    print(string.format('loading %d images (%dx%d)', count, rows, cols))

    -- read all pixel data
    local pixels = rows * cols
    local data = Tensor.new({count, pixels})

    for i = 0, count - 1 do
        local raw = f:read(pixels)
        for j = 1, pixels do
            -- normalize to 0-1 range
            data:set(i * pixels + (j - 1), string.byte(raw, j) / 255.0)
        end
    end

    f:close()
    return data
end

local function read_mnist_labels(path)
    local f = io.open(path, 'rb')
    if not f then error('could not open: ' .. path) end

    -- header: magic(4), count(4)
    local header = f:read(8)
    local magic  = string.unpack('>I4', header, 1)
    local count  = string.unpack('>I4', header, 5)

    print(string.format('loading %d labels', count))

    local labels = Tensor.new({count})
    local raw = f:read(count)
    for i = 1, count do
        labels:set(i - 1, string.byte(raw, i))
    end

    f:close()
    return labels
end

-- download mnist if not already present
local function download_mnist(dir)
    os.execute('mkdir -p ' .. dir)

    local files = {
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte',
    }

    local base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

    for _, name in ipairs(files) do
        local path = dir .. '/' .. name
        local f = io.open(path, 'r')
        if f then
            f:close()
        else
            local gz_path = path .. '.gz'
            print('downloading ' .. name .. '...')
            os.execute(string.format('curl -sL %s%s.gz -o %s', base_url, name, gz_path))
            os.execute(string.format('gunzip -f %s', gz_path))
        end
    end
end

-- main
local data_dir = 'data/mnist'
download_mnist(data_dir)

-- load data
local train_images = read_mnist_images(data_dir .. '/train-images-idx3-ubyte')
local train_labels = read_mnist_labels(data_dir .. '/train-labels-idx1-ubyte')
local test_images  = read_mnist_images(data_dir .. '/t10k-images-idx3-ubyte')
local test_labels  = read_mnist_labels(data_dir .. '/t10k-labels-idx1-ubyte')

-- build network
-- 784 input (28x28 pixels) -> 256 -> 128 -> 10 classes
local model = Sequential.new(
    Linear.new(784, 256),
    activation.ReLU.new(),
    Dropout.new(0.2),
    Linear.new(256, 128),
    activation.ReLU.new(),
    Dropout.new(0.2),
    Linear.new(128, 10)
)

print(tostring(model))
print(string.format('total parameters: %d', model:num_params()))

-- setup
local params    = model:parameters()
local optimizer = Adam.new(params, 0.001)
local criterion = loss_fn.CrossEntropyLoss.new()

-- data loaders
local train_loader = DataLoader.new(train_images, train_labels, 64)
local test_loader  = DataLoader.new(test_images, test_labels, 256, false)

-- compute accuracy on a dataset
local function evaluate(loader)
    -- disable dropout for evaluation
    for _, layer in ipairs(model.layers) do
        if layer.eval then layer:eval() end
    end

    autograd.enabled = false

    local correct = 0
    local total   = 0

    for batch_data, batch_targets in loader:iter() do
        local pred = model(batch_data)
        local bs   = batch_data.shape[1]

        -- find predicted class (argmax) for each sample
        for b = 0, bs - 1 do
            local max_val = pred:get(b * 10)
            local max_idx = 0
            for c = 1, 9 do
                local val = pred:get(b * 10 + c)
                if val > max_val then
                    max_val = val
                    max_idx = c
                end
            end

            local target = math.floor(batch_targets:get(b))
            if max_idx == target then
                correct = correct + 1
            end
            total = total + 1
        end
    end

    autograd.enabled = true

    -- re-enable dropout for training
    for _, layer in ipairs(model.layers) do
        if layer.train then layer:train() end
    end

    return correct / total
end

-- training loop
local epochs = 10
print(string.format('\ntraining for %d epochs...', epochs))

for epoch = 1, epochs do
    local epoch_loss = 0
    local n_batches  = 0

    for batch_data, batch_targets in train_loader:iter() do
        -- clear state
        autograd.zero_graph()
        model:zero_grad()

        -- forward
        autograd.watch(batch_data)
        local pred = model(batch_data)
        local loss = criterion(pred, batch_targets)

        -- backward
        autograd.backward(loss)

        -- update weights
        optimizer:step()

        epoch_loss = epoch_loss + loss:get(0)
        n_batches  = n_batches + 1
    end

    local avg_loss = epoch_loss / n_batches
    local accuracy = evaluate(test_loader)

    print(string.format('epoch %2d  loss: %.4f  test accuracy: %.2f%%',
        epoch, avg_loss, accuracy * 100))
end

-- save trained model
checkpoint.save(model, 'mnist_model.bin')
print('\nmodel saved to mnist_model.bin')

-- final test
local final_acc = evaluate(test_loader)
print(string.format('final test accuracy: %.2f%%', final_acc * 100))
