#!/usr/bin/env python3

import numpy as np


def init_layer(img):
    """
    Concatenate img features across color channels into (H*W*C, 1). Then normalize.
    """
    w, h, c = img.shape
    layer = None
    for i in range(c):
        temp = img[:, :, i].reshape(w * h, 1)
        if layer is None:
            layer = temp
        else:
            layer = np.concatenate((layer, temp), axis=0)
    return layer / 255


def img2col_with_stride(data_im, channels, height, width, ksize, stride, pad):
    """
    Args:
        data_im: image data in shape (H*W*C, 1).
        channels: num of input feature depth.
        height: image height.
        width: image width.
        ksize: kernel width and height.
        stride: convolution stride step.
        pad: padding length.
    """
    height_col = (height + 2 * pad - ksize) // stride + 1
    width_col = (width + 2 * pad - ksize) // stride + 1
    channels_col = channels * ksize * ksize
    data_col = np.zeros(shape=(height_col * width_col * channels_col, 1))
    for c in range(channels_col):
        w_offset = np.mod(c, ksize)
        h_offset = np.mod(np.floor(c // ksize), ksize)
        c_im = np.floor(np.floor(c // ksize) // ksize)
        for h in range(height_col):
            for w in range(width_col):
                im_row = h_offset + h * stride
                im_col = w_offset + w * stride
                col_index = (c * height_col + h) * width_col + w
                data_col[col_index] = im2col_get_pixel(
                    data_im, height, width, im_row, im_col, c_im, pad
                )
    return data_col


def im2col_get_pixel(im, height, width, row, col, channel, pad):
    """
    Args:
        im: input image.
        height: image height.
        width: image width.
        row: row index.
        col: col index.
        channel: channel index.
        pad: padding length.
    """
    row = row - pad
    col = col - pad
    if row < 0 or col < 0 or row >= height or col >= width:
        pixel = 0
    else:
        pixel = im[int(col + width * (row + height * channel))]
    return pixel


def forward_convolution(
    layer_in,
    size_in,
    kernel_size,
    maps_in,
    maps_out,
    # norm,
    weights,
    # biases,
    # scales,
    # rolling_mean,
    # rolling_variance,
):
    """
    Args:
        layer_in: input feature map in shape (H*W*C, 1).
        size_in: shape (H, W, C)
        kernel_size: kernel size.
        maps_in: input feature map depth.
        maps_out: output feature map depth.
    """
    width = size_in[1]
    height = size_in[0]
    size_out = [width, height]
    stride = 1
    pad = (kernel_size - 1) / 2
    data_col = img2col(layer_in, maps_in, height, width, kernel_size, stride, pad)
    m = maps_out
    k = kernel_size * kernel_size * maps_in
    n = width * height
    A = weights.reshape(m, k)
    B = data_col.reshape(k, n)
    C = np.dot(A, B)
    layer_out = C.reshape(m * n, 1)
    # if norm:
    #     for i in range(maps_out):
    #         for j in range(width * height):
    #             index = i * width * height + j
    #             # normalization
    #             layer_out[index] = (layer_out[index] - rolling_mean[i]) / (
    #                 np.sqrt(rolling_variance[i]) + 0.000001
    #             )
    #             # scale and bias
    #             layer_out[index] = layer_out[index] * scales[i]
    #             layer_out[index] = layer_out[index] + biases[i]
    #             # activation
    #             if layer_out[index] < 0:
    #                 layer_out[index] = layer_out[index] * 0.1
    # else:
    #     for i in range(maps_out):
    #         for j in range(width * height):
    #             index = i * width * height + j
    #             layer_out[index] = layer_out[index] + biases[i]
    print(
        "Done! %d*%d*%d=>%d*%d*%d" % (width, height, maps_in, width, height, maps_out)
    )
    return layer_out, size_out


img_shape = (5, 5, 3)
h, w, c = img_shape
img = np.arange(h * w * c).reshape(img_shape)
data_im = init_layer(img)
flattened_img = img2col_with_stride(data_im, 3, 5, 5, 3, 1, 1)
assert flattened_img.shape == (5 * 5 * 3 * 3 * 3, 1)
