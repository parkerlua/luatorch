#include "../tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// im2col: unrolls image patches into columns for matmul-based convolution
// this is the core operation that makes conv2d fast
// input shape: [batch, channels, height, width] stored flat
// output shape: [batch * out_h * out_w, channels * kh * kw]
// perf fix: moved from pure Lua nested loops to C, ~100x faster for real images
Tensor* tensor_im2col(Tensor* input, int batch, int channels, int height, int width,
                       int kernel_size, int stride, int padding) {
    if (!input) return NULL;

    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width  + 2 * padding - kernel_size) / stride + 1;
    int col_len = channels * kernel_size * kernel_size;
    int64_t total_rows = (int64_t)batch * out_h * out_w;

    int64_t out_shape[2] = {total_rows, col_len};
    Tensor* cols = tensor_new(out_shape, 2, input->dtype, input->device);
    if (!cols) return NULL;

    for (int b = 0; b < batch; b++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int64_t row = (int64_t)(b * out_h * out_w + oh * out_w + ow);
                int col = 0;
                for (int c = 0; c < channels; c++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;
                            float val = 0.0f;
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int64_t idx = ((int64_t)(b * channels + c) * height + ih) * width + iw;
                                val = input->data[idx];
                            }
                            cols->data[row * col_len + col] = val;
                            col++;
                        }
                    }
                }
            }
        }
    }

    return cols;
}
