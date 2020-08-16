#!/usr/bin/env python3

import copy
import time

import numpy as np


def conv2d(x, f):
    r, c = x.shape
    kr, kc = f.shape
    res = np.zeros(x.shape)
    for i in range(r):
        for j in range(c):
            for mm in range(kr - 1, -1, -1):
                for nn in range(kc - 1, -1, -1):
                    ii = i + kr // 2 - mm
                    jj = j + kc // 2 - nn
                    if 0 <= ii < r and 0 <= jj < c:
                        res[i][j] += x[ii][jj] * f[mm][nn]
    return res


start = time.time()
img = copy.deepcopy(image)
end = time.time()
print("Finish in %.2f ms" % ((end - start) * 1000))
print("my_res = ", res)
