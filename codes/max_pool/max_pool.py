#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided


def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    """
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """
    # Padding
    A = np.pad(A, padding, mode="constant")

    # Window view of A
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )
    kernel_size = (kernel_size, kernel_size)
    # numpy strides are offsets of the axises, since the dtype is np.int64
    # each element take 8 bytes
    assert A.strides == (32, 8)
    A_w = as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides,
    )
    # Create a strides view with smaller strides steps in the first two axises.
    assert A_w.strides == (64, 16, 32, 8)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


A = np.array([[1, 1, 2, 4], [5, 6, 7, 8], [3, 2, 1, 0], [1, 2, 3, 4]], dtype=np.int64)

print(pool2d(A, kernel_size=2, stride=2, padding=0, pool_mode="max"))


def stride_1d(line_in, L, W, D, stride):
    window = as_strided(
        line_in, shape=((L - W) // stride + 1, W), strides=(stride * D, D)
    )
    # plt.plot(window)
    # plt.show()
    return window


L = 100  # sample count
W = 10  # window width
D = 8  # size of the datum in type, dtype np.int64 uses 8 bytes
signal = np.sin(np.linspace(0, 20, L))
noise = np.random.uniform(-0.1, 0.1, L)
line_in = signal + noise
stride_1d(line_in, L, W, D, 2)

L = 8
B = np.arange(L, dtype=np.int32)
window = stride_1d(B, L, 2, 4, 2)
np.testing.assert_allclose(
    window, np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int32)
)
