local Tensor   = require('luatorch.tensor')
local autograd = require('luatorch.autograd')

-- loss functions for training neural networks
-- each one computes the loss and registers autograd for backward pass

-- mean squared error loss
-- the default choice for regression
local MSELoss = {}
MSELoss.__index = MSELoss

function MSELoss.new()
    return setmetatable({}, MSELoss)
end

function MSELoss:forward(pred, target)
    local loss = Tensor.mse_loss(pred, target)

    autograd.record("mse_loss", {pred, target}, loss, function(grad)
        if pred.requires_grad then
            autograd.acc_grad(pred, Tensor.mse_loss_backward(pred, target))
        end
    end)

    return loss
end

MSELoss.__call = function(self, pred, target)
    return self:forward(pred, target)
end

function MSELoss:__tostring() return 'MSELoss()' end

-- mean absolute error loss
-- more robust to outliers than mse
local MAELoss = {}
MAELoss.__index = MAELoss

function MAELoss.new()
    return setmetatable({}, MAELoss)
end

function MAELoss:forward(pred, target)
    local loss = Tensor.mae_loss(pred, target)

    autograd.record("mae_loss", {pred, target}, loss, function(grad)
        if pred.requires_grad then
            autograd.acc_grad(pred, Tensor.mae_loss_backward(pred, target))
        end
    end)

    return loss
end

MAELoss.__call = function(self, pred, target)
    return self:forward(pred, target)
end

function MAELoss:__tostring() return 'MAELoss()' end

-- cross entropy loss
-- the standard loss for classification
-- takes raw logits and class indices
local CrossEntropyLoss = {}
CrossEntropyLoss.__index = CrossEntropyLoss

function CrossEntropyLoss.new()
    return setmetatable({}, CrossEntropyLoss)
end

function CrossEntropyLoss:forward(pred, target)
    local loss = Tensor.cross_entropy_loss(pred, target)

    autograd.record("cross_entropy_loss", {pred, target}, loss, function(grad)
        if pred.requires_grad then
            autograd.acc_grad(pred, Tensor.cross_entropy_loss_backward(pred, target))
        end
    end)

    return loss
end

CrossEntropyLoss.__call = function(self, pred, target)
    return self:forward(pred, target)
end

function CrossEntropyLoss:__tostring() return 'CrossEntropyLoss()' end

return {
    MSELoss          = MSELoss,
    MAELoss          = MAELoss,
    CrossEntropyLoss = CrossEntropyLoss,
}
