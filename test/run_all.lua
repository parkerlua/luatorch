-- test runner
-- runs every test file, prints pass/fail, exits with error code if anything fails

local test_files = {
    'test/test_tensor.lua',
    'test/test_arithmetic.lua',
    'test/test_matmul.lua',
    'test/test_autograd.lua',
    'test/test_nn.lua',
    'test/test_optim.lua',
    'test/test_training.lua',
    'test/test_cuda.lua',
}

local passed = 0
local failed = 0
local skipped = 0
local errors = {}

print('luatorch test suite')
print(string.rep('=', 60))

for _, file in ipairs(test_files) do
    io.write(string.format('%-40s ', file))
    io.flush()

    local ok, err = pcall(dofile, file)
    if ok then
        print('PASS')
        passed = passed + 1
    else
        if err and err:match('SKIP') then
            print('SKIP')
            skipped = skipped + 1
        else
            print('FAIL')
            failed = failed + 1
            table.insert(errors, {file = file, err = tostring(err)})
        end
    end
end

print(string.rep('=', 60))
print(string.format('passed: %d  failed: %d  skipped: %d', passed, failed, skipped))

if #errors > 0 then
    print('\nfailure details:')
    for _, e in ipairs(errors) do
        print(string.format('  %s: %s', e.file, e.err))
    end
end

if failed > 0 then
    os.exit(1)
end
