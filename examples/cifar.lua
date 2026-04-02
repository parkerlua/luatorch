-- cifar-10 image classification
-- demonstrates conv2d + batchnorm + relu
-- downloads cifar-10, trains for 5 epochs

local lf = require('luatorch')

local Tensor      = lf.Tensor
local autograd    = lf.autograd
local nn          = lf.nn
local Adam        = lf.optim.Adam
local DataLoader  = lf.data.DataLoader
local Logger      = lf.utils.Logger
local checkpoint  = lf.io.checkpoint

-- cifar-10 binary file reader
-- each file has 10000 images, each is 1 byte label + 3072 bytes pixels (32x32x3)
local function read_cifar_batch(path)
    local f = io.open(path, 'rb')
    if not f then error('could not open: ' .. path) end

    local n_samples = 10000
    local img_size  = 3072  -- 3 * 32 * 32

    local images = Tensor.new({n_samples, img_size})
    local labels = Tensor.new({n_samples})

    for i = 0, n_samples - 1 do
        -- first byte is label
        local label_byte = f:read(1)
        labels:set(i, string.byte(label_byte))

        -- next 3072 bytes are the image (RGB planes)
        local raw = f:read(img_size)
        for j = 1, img_size do
            images:set(i * img_size + (j - 1), string.byte(raw, j) / 255.0)
        end
    end

    f:close()
    return images, labels
end

-- download cifar-10 if not present
local data_dir = 'data/cifar-10'
os.execute('mkdir -p ' .. data_dir)

local test_file = data_dir .. '/test_batch.bin'
local f = io.open(test_file, 'r')
if f then
    f:close()
else
    print('downloading cifar-10...')
    os.execute(string.format(
        'curl -sL https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz | tar -xz -C data/ && ' ..
        'mv data/cifar-10-batches-bin/* %s/ && rmdir data/cifar-10-batches-bin',
        data_dir))
end

-- load training data (5 batches)
print('loading cifar-10...')
local train_images_list = {}
local train_labels_list = {}

for batch_num = 1, 5 do
    local path = string.format('%s/data_batch_%d.bin', data_dir, batch_num)
    local imgs, lbls = read_cifar_batch(path)
    table.insert(train_images_list, imgs)
    table.insert(train_labels_list, lbls)
end

-- combine into single tensors
local total_train = 50000
local train_images = Tensor.new({total_train, 3072})
local train_labels = Tensor.new({total_train})

for b = 0, 4 do
    for i = 0, 9999 do
        local global_i = b * 10000 + i
        train_labels:set(global_i, train_labels_list[b + 1]:get(i))
        for j = 0, 3071 do
            train_images:set(global_i * 3072 + j, train_images_list[b + 1]:get(i * 3072 + j))
        end
    end
end

-- load test data
local test_images, test_labels = read_cifar_batch(data_dir .. '/test_batch.bin')

print(string.format('train: %d images, test: %d images', total_train, 10000))

-- build a simple network
-- flatten the images and use MLP since conv is slow on cpu
-- for a real run, use Conv2d on the 4090
local model = nn.Sequential(
    nn.Linear(3072, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10)
)

print(tostring(model))
print(string.format('total parameters: %d', model:num_params()))

-- setup
local params    = model:parameters()
local optimizer = Adam.new(params, 0.001)
local criterion = nn.CrossEntropyLoss.new()
local logger    = Logger.new(50)

-- data loaders
local train_loader = DataLoader.new(train_images, train_labels, 64)
local test_loader  = DataLoader.new(test_images, test_labels, 256, false)

-- compute accuracy
local function evaluate(loader)
    for _, layer in ipairs(model.layers) do
        if layer.eval then layer:eval() end
    end

    autograd.enabled = false

    local correct = 0
    local total   = 0

    for batch_data, batch_targets in loader:iter() do
        local pred = model(batch_data)
        local bs   = batch_data.shape[1]

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
            if max_idx == math.floor(batch_targets:get(b)) then
                correct = correct + 1
            end
            total = total + 1
        end
    end

    autograd.enabled = true
    for _, layer in ipairs(model.layers) do
        if layer.train then layer:train() end
    end

    return correct / total
end

-- training loop
local epochs = 5
print(string.format('\ntraining for %d epochs...', epochs))

for epoch = 1, epochs do
    logger:set_epoch(epoch)
    local epoch_loss = 0
    local n_batches  = 0

    for batch_data, batch_targets in train_loader:iter() do
        autograd.zero_graph()
        model:zero_grad()

        autograd.watch(batch_data)
        local pred = model(batch_data)
        local loss = criterion(pred, batch_targets)

        autograd.backward(loss)
        optimizer:step()

        local loss_val = loss:get(0)
        epoch_loss = epoch_loss + loss_val
        n_batches  = n_batches + 1

        logger:log(loss_val, nil, optimizer.lr)
    end

    local avg_loss  = epoch_loss / n_batches
    local train_acc = evaluate(train_loader)
    local test_acc  = evaluate(test_loader)

    print(string.format('epoch %d  loss: %.4f  train acc: %.2f%%  test acc: %.2f%%',
        epoch, avg_loss, train_acc * 100, test_acc * 100))
end

-- save model
checkpoint.save(model, 'cifar_model.bin')
print('\nmodel saved to cifar_model.bin')

-- save log
logger:save_csv('cifar_training.csv')

local final_acc = evaluate(test_loader)
print(string.format('final test accuracy: %.2f%%', final_acc * 100))
