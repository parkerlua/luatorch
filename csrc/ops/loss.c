#include "../tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// mean squared error
// mse = mean((pred - target)^2)
// the go to loss for regression problems
// returns a scalar tensor of shape [1]
Tensor* tensor_mse_loss(Tensor* pred, Tensor* target) {
    if (!pred || !target) return NULL;
    if (pred->numel != target->numel) {
        fprintf(stderr, "luatorch error: mse size mismatch\n");
        return NULL;
    }

    float sum = 0.0f;
    for (int64_t i = 0; i < pred->numel; i++) {
        float diff = pred->data[i] - target->data[i];
        sum += diff * diff;
    }

    int64_t out_shape[1] = {1};
    Tensor* out = tensor_new(out_shape, 1, pred->dtype, pred->device);
    if (!out) return NULL;
    out->data[0] = sum / (float)pred->numel;
    return out;
}

// gradient of mse loss
// d(mse)/d(pred) = 2 * (pred - target) / n
Tensor* tensor_mse_loss_backward(Tensor* pred, Tensor* target) {
    if (!pred || !target) return NULL;
    if (pred->numel != target->numel) {
        fprintf(stderr, "luatorch error: mse backward size mismatch\n");
        return NULL;
    }

    Tensor* grad = tensor_new(pred->shape, pred->ndim, pred->dtype, pred->device);
    if (!grad) return NULL;

    float scale = 2.0f / (float)pred->numel;
    for (int64_t i = 0; i < pred->numel; i++) {
        grad->data[i] = scale * (pred->data[i] - target->data[i]);
    }
    return grad;
}

// mean absolute error
// mae = mean(|pred - target|)
// more robust to outliers than mse
Tensor* tensor_mae_loss(Tensor* pred, Tensor* target) {
    if (!pred || !target) return NULL;
    if (pred->numel != target->numel) {
        fprintf(stderr, "luatorch error: mae size mismatch\n");
        return NULL;
    }

    float sum = 0.0f;
    for (int64_t i = 0; i < pred->numel; i++) {
        sum += fabsf(pred->data[i] - target->data[i]);
    }

    int64_t out_shape[1] = {1};
    Tensor* out = tensor_new(out_shape, 1, pred->dtype, pred->device);
    if (!out) return NULL;
    out->data[0] = sum / (float)pred->numel;
    return out;
}

// gradient of mae loss
// d(mae)/d(pred) = sign(pred - target) / n
Tensor* tensor_mae_loss_backward(Tensor* pred, Tensor* target) {
    if (!pred || !target) return NULL;

    Tensor* grad = tensor_new(pred->shape, pred->ndim, pred->dtype, pred->device);
    if (!grad) return NULL;

    float scale = 1.0f / (float)pred->numel;
    for (int64_t i = 0; i < pred->numel; i++) {
        float diff = pred->data[i] - target->data[i];
        if (diff > 0.0f) grad->data[i] = scale;
        else if (diff < 0.0f) grad->data[i] = -scale;
        else grad->data[i] = 0.0f;
    }
    return grad;
}

// cross entropy loss
// pred should be log probabilities (after log_softmax)
// target is class indices stored as floats
// loss = -sum(log_prob[target_class]) / batch_size
// this is the standard loss for classification
Tensor* tensor_cross_entropy_loss(Tensor* pred, Tensor* target) {
    if (!pred || !target) return NULL;

    // pred is [batch, classes], target is [batch]
    if (pred->ndim != 2 || target->ndim != 1) {
        fprintf(stderr, "luatorch error: cross entropy expects pred [batch, classes] and target [batch]\n");
        return NULL;
    }

    int64_t batch   = pred->shape[0];
    int64_t classes = pred->shape[1];

    if (target->numel != batch) {
        fprintf(stderr, "luatorch error: cross entropy batch size mismatch\n");
        return NULL;
    }

    // first compute softmax of pred for numerical stability
    // then take negative log of the correct class probability
    float total_loss = 0.0f;
    for (int64_t b = 0; b < batch; b++) {
        float* row = pred->data + b * classes;
        int64_t label = (int64_t)target->data[b];

        // fix: bounds check label to prevent out of bounds read
        if (label < 0 || label >= classes) {
            fprintf(stderr, "luatorch error: cross entropy label %lld out of range [0, %lld)\n",
                (long long)label, (long long)classes);
            continue;
        }

        // find max for numerical stability
        float max_val = row[0];
        for (int64_t c = 1; c < classes; c++) {
            if (row[c] > max_val) max_val = row[c];
        }

        // compute log(sum(exp(x - max)))
        float log_sum_exp = 0.0f;
        for (int64_t c = 0; c < classes; c++) {
            log_sum_exp += expf(row[c] - max_val);
        }
        log_sum_exp = logf(log_sum_exp) + max_val;

        // loss for this sample is -(pred[label] - log_sum_exp)
        total_loss += -(row[label] - log_sum_exp);
    }

    int64_t out_shape[1] = {1};
    Tensor* out = tensor_new(out_shape, 1, pred->dtype, pred->device);
    if (!out) return NULL;
    out->data[0] = total_loss / (float)batch;
    return out;
}

// gradient of cross entropy loss
// grad = softmax(pred) - one_hot(target)
// this is the beautifully simple gradient of softmax + cross entropy
Tensor* tensor_cross_entropy_loss_backward(Tensor* pred, Tensor* target) {
    if (!pred || !target) return NULL;

    int64_t batch   = pred->shape[0];
    int64_t classes = pred->shape[1];

    Tensor* grad = tensor_new(pred->shape, pred->ndim, pred->dtype, pred->device);
    if (!grad) return NULL;

    for (int64_t b = 0; b < batch; b++) {
        float* row      = pred->data + b * classes;
        float* grad_row = grad->data + b * classes;
        int64_t label   = (int64_t)target->data[b];

        // fix: bounds check label
        if (label < 0 || label >= classes) {
            for (int64_t c = 0; c < classes; c++) grad_row[c] = 0.0f;
            continue;
        }

        // compute softmax for this row
        float max_val = row[0];
        for (int64_t c = 1; c < classes; c++) {
            if (row[c] > max_val) max_val = row[c];
        }
        float sum = 0.0f;
        for (int64_t c = 0; c < classes; c++) {
            grad_row[c] = expf(row[c] - max_val);
            sum += grad_row[c];
        }
        for (int64_t c = 0; c < classes; c++) {
            grad_row[c] /= sum;
        }

        // subtract 1 from the correct class
        grad_row[label] -= 1.0f;

        // average over batch
        for (int64_t c = 0; c < classes; c++) {
            grad_row[c] /= (float)batch;
        }
    }

    return grad;
}
