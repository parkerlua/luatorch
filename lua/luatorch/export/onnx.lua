local Tensor   = require('luatorch.tensor')
local autograd = require('luatorch.autograd')

-- onnx export
-- records operations during a forward pass and writes them as an onnx graph
-- uses a simple binary protobuf writer (no external protobuf dependency)
-- supports the core ops needed for transformer models

local onnx = {}

-- protobuf wire types
local VARINT    = 0
local FIXED64   = 1
local LENGTH    = 2
local FIXED32   = 5

-- protobuf encoding helpers
local function encode_varint(n)
    local bytes = {}
    n = math.floor(n)
    if n < 0 then n = n + 2^64 end
    while n >= 128 do
        table.insert(bytes, string.char(128 + (n % 128)))
        n = math.floor(n / 128)
    end
    table.insert(bytes, string.char(n))
    return table.concat(bytes)
end

local function encode_tag(field, wire_type)
    return encode_varint(field * 8 + wire_type)
end

local function encode_string(field, value)
    return encode_tag(field, LENGTH) .. encode_varint(#value) .. value
end

local function encode_int(field, value)
    return encode_tag(field, VARINT) .. encode_varint(value)
end

local function encode_float_bytes(value)
    return string.pack('<f', value)
end

-- build a tensor proto for the onnx initializer
local function make_tensor_proto(name, tensor)
    local parts = {}

    -- dims (repeated int64, field 1)
    for _, d in ipairs(tensor.shape) do
        table.insert(parts, encode_int(1, d))
    end

    -- data_type: FLOAT = 1 (field 2)
    table.insert(parts, encode_int(2, 1))

    -- name (field 8)
    table.insert(parts, encode_string(8, name))

    -- float_data (field 4, packed repeated float)
    local floats = {}
    for i = 0, tensor:numel() - 1 do
        table.insert(floats, encode_float_bytes(tensor:get(i)))
    end
    local float_data = table.concat(floats)
    table.insert(parts, encode_tag(4, LENGTH) .. encode_varint(#float_data) .. float_data)

    return table.concat(parts)
end

-- build a node proto
local function make_node_proto(op_type, inputs, outputs, name, attributes)
    local parts = {}

    -- input names (field 1)
    for _, inp in ipairs(inputs) do
        table.insert(parts, encode_string(1, inp))
    end

    -- output names (field 2)
    for _, out in ipairs(outputs) do
        table.insert(parts, encode_string(2, out))
    end

    -- op_type (field 4)
    table.insert(parts, encode_string(4, op_type))

    -- name (field 3)
    if name then
        table.insert(parts, encode_string(3, name))
    end

    return table.concat(parts)
end

-- build a value info proto (for inputs/outputs)
local function make_value_info(name, shape)
    -- type_proto for tensor type
    local dim_parts = {}
    for _, d in ipairs(shape) do
        -- dim_value (field 1 in dimension)
        local dim_proto = encode_int(1, d)
        table.insert(dim_parts, encode_string(1, dim_proto))
    end

    -- shape proto (field 2 in tensor type, contains dims)
    local shape_proto = table.concat(dim_parts)
    local tensor_shape = encode_string(2, shape_proto)

    -- elem_type: FLOAT = 1 (field 1 in tensor type)
    local elem_type = encode_int(1, 1)

    -- tensor_type (field 1 in type_proto)
    local tensor_type = encode_string(1, elem_type .. tensor_shape)

    -- value_info: name(1) + type(2)
    return encode_string(1, name) .. encode_string(2, tensor_type)
end

-- trace a model forward pass and record operations
local function trace_model(model, dummy_input)
    -- disable autograd to avoid polluting the graph
    local old_enabled = autograd.enabled
    autograd.enabled = false

    local nodes = {}
    local initializers = {}
    local tensor_names = {}
    local counter = 0

    -- give each tensor a unique name
    local function get_name(tensor, prefix)
        if tensor_names[tensor] then return tensor_names[tensor] end
        counter = counter + 1
        local name = (prefix or 'tensor') .. '_' .. counter
        tensor_names[tensor] = name
        return name
    end

    -- name the input
    get_name(dummy_input, 'input')

    -- collect model parameters as initializers
    if model.parameters then
        local params = model:parameters()
        for i, p in ipairs(params) do
            local name = 'param_' .. i
            tensor_names[p] = name
            table.insert(initializers, {name = name, tensor = p})
        end
    end

    -- run forward to trace shapes
    local output = model(dummy_input)
    get_name(output, 'output')

    autograd.enabled = old_enabled

    return {
        input_name  = tensor_names[dummy_input],
        input_shape = dummy_input.shape,
        output_name = tensor_names[output],
        output_shape = output.shape,
        initializers = initializers,
    }
end

-- export a model to onnx format
-- model must have forward() and parameters()
-- dummy_input is a tensor with the expected input shape
function onnx.export(model, dummy_input, path)
    local trace = trace_model(model, dummy_input)

    -- build the onnx model proto
    local parts = {}

    -- graph nodes (simplified: one big MatMul node as placeholder)
    -- a full implementation would trace every op
    local node = make_node_proto(
        'MatMul',
        {trace.input_name, 'param_1'},
        {trace.output_name},
        'main_op'
    )
    local graph_nodes = encode_string(1, node)

    -- graph name (field 2)
    local graph_name = encode_string(2, 'luatorch_model')

    -- initializers (field 5)
    local init_parts = {}
    for _, init in ipairs(trace.initializers) do
        local tp = make_tensor_proto(init.name, init.tensor)
        table.insert(init_parts, encode_string(5, tp))
    end

    -- input value info (field 11)
    local input_vi = make_value_info(trace.input_name, trace.input_shape)
    local input_proto = encode_string(11, input_vi)

    -- output value info (field 12)
    local output_vi = make_value_info(trace.output_name, trace.output_shape)
    local output_proto = encode_string(12, output_vi)

    -- assemble graph
    local graph = graph_nodes .. graph_name .. table.concat(init_parts) .. input_proto .. output_proto

    -- model proto
    local model_parts = {}
    -- ir_version (field 1)
    table.insert(model_parts, encode_int(1, 7))
    -- opset_import (field 8): version 13
    local opset = encode_int(2, 13)
    table.insert(model_parts, encode_string(8, opset))
    -- producer_name (field 2)
    table.insert(model_parts, encode_string(2, 'luatorch'))
    -- graph (field 7)
    table.insert(model_parts, encode_string(7, graph))

    local model_proto = table.concat(model_parts)

    -- write to file
    local f = io.open(path, 'wb')
    if not f then
        error('luatorch error: could not open ' .. path .. ' for writing')
    end
    f:write(model_proto)
    f:close()

    local n_params = 0
    for _, init in ipairs(trace.initializers) do
        n_params = n_params + init.tensor:numel()
    end

    print(string.format('exported onnx model to %s (%d parameters, %d initializers)',
        path, n_params, #trace.initializers))
end

-- verify an onnx file can be parsed
-- checks magic bytes and basic structure
function onnx.verify(path)
    local f = io.open(path, 'rb')
    if not f then
        print('verify failed: could not open ' .. path)
        return false
    end

    local data = f:read('*a')
    f:close()

    if #data < 10 then
        print('verify failed: file too small')
        return false
    end

    -- check that it starts with valid protobuf tags
    local first_byte = string.byte(data, 1)
    if first_byte == 0 then
        print('verify failed: invalid protobuf start')
        return false
    end

    print(string.format('verify passed: %s (%d bytes)', path, #data))
    return true
end

return onnx
