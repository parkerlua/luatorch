local Tensor = require('luatorch.tensor')

-- autograd tracks every operation done on a tensor
-- then walks backwards through them to compute gradients

local Node = {}
Node.__index = Node

function Node.new(tensor, op, parents)
    local self = setmetatable({}, Node)
    self.tensor   = tensor
    self.op       = op
    self.parents  = parents or {}
    self.grad_fn  = nil
    return self
end

local autograd = {}

autograd.enabled = true
autograd.graph = {}

function autograd.zero_graph()
    autograd.graph = {}
end

function autograd.watch(tensor)
    tensor.requires_grad = true
    tensor.grad          = nil
    tensor._node         = Node.new(tensor, "leaf", {})
    return tensor
end

function autograd.record(op, inputs, output, grad_fn)
    if not autograd.enabled then return end

    local node = Node.new(output, op, inputs)
    node.grad_fn = grad_fn
    output._node = node

    table.insert(autograd.graph, node)
end

-- accumulate gradient: if grad already exists, add to it
-- this is critical for tensors used more than once in the graph
-- (e.g. residual connections, weight sharing)
local function acc_grad(tensor, new_grad)
    if tensor.grad then
        tensor.grad = Tensor.add(tensor.grad, new_grad)
    else
        tensor.grad = new_grad
    end
end

-- addition: out = a + b
-- backward: both inputs get the gradient unchanged
function autograd.add(a, b)
    local out = Tensor.add(a, b)

    autograd.record("add", {a, b}, out, function(grad)
        if a.requires_grad then
            acc_grad(a, grad)
        end
        if b.requires_grad then
            acc_grad(b, grad)
        end
    end)

    return out
end

-- multiplication: out = a * b
-- backward: grad_a = grad * b, grad_b = grad * a
function autograd.mul(a, b)
    local out = Tensor.mul(a, b)

    autograd.record("mul", {a, b}, out, function(grad)
        if a.requires_grad then
            acc_grad(a, Tensor.mul(grad, b))
        end
        if b.requires_grad then
            acc_grad(b, Tensor.mul(grad, a))
        end
    end)

    return out
end

-- matmul: out = a @ b
-- backward: grad_a = grad @ b.T, grad_b = a.T @ grad
function autograd.matmul(a, b)
    local out = Tensor.matmul(a, b)

    autograd.record("matmul", {a, b}, out, function(grad)
        if a.requires_grad then
            local b_t = Tensor.transpose(b)
            acc_grad(a, Tensor.matmul(grad, b_t))
        end
        if b.requires_grad then
            local a_t = Tensor.transpose(a)
            acc_grad(b, Tensor.matmul(a_t, grad))
        end
    end)

    return out
end

-- relu: out = max(0, x)
-- backward: grad where x > 0, zero where x <= 0
function autograd.relu(x)
    local out = Tensor.relu(x)

    autograd.record("relu", {x}, out, function(grad)
        if x.requires_grad then
            local mask = Tensor.gt_scalar(x, 0.0)
            acc_grad(x, Tensor.mul(grad, mask))
        end
    end)

    return out
end

-- backward pass, walks graph in reverse
function autograd.backward(loss)
    local ones_shape = loss.shape
    local seed_grad  = Tensor.new(ones_shape)
    seed_grad:ones()

    loss.grad = seed_grad

    for i = #autograd.graph, 1, -1 do
        local node = autograd.graph[i]
        if node.grad_fn and node.tensor.grad then
            node.grad_fn(node.tensor.grad)
        end
    end
end

function autograd.zero_grad(tensors)
    for _, t in ipairs(tensors) do
        t.grad = nil
    end
end

-- expose acc_grad for use by other modules
autograd.acc_grad = acc_grad

return autograd
