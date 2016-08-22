#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
import matplotlib.pyplot as plt

__author__ = 'fyabc'


def plotMatrix(matrix):
    plt.imshow(matrix, cmap='gray', vmin=0, vmax=1)
    plt.show()


def test():
    import numpy as np
    mat = np.random.randn(100, 100)
    plotMatrix(mat)


if __name__ == '__main__':
    test()
