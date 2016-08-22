#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import numpy as np
import theano.tensor as T
from theano import shared, config, function

__author__ = 'fyabc'

fX = config.floatX


def toFX(value):
    return eval('%s(value)' % fX)


class PolicyNetwork(object):
    """
    A simple policy network.

    action = sigmoid(W * state + b)

    H: output size of NN
    B: batch size, the size of mask array

    state:  (H + B)
    W:      (B, H + B)
    b:      (B)
    state:  (B)
    """

    def __init__(self, H, B):
        self.H = H
        self.B = B

        self.W = shared(name='W', value=self.initNorm(self.B, self.H + self.B))
        self.b = shared(name='b', value=self.initNorm(self.B))
        self.parameters = [self.W, self.b]

        self.state = T.vector(name='state', dtype=fX)
        self.action = T.nnet.sigmoid(T.dot(self.W, self.state) + self.b)

        self.actionFunction = function(
            inputs=[self.state],
            outputs=self.action,
        )

    @staticmethod
    def initNorm(*dims):
        return np.asarray(np.random.randn(*dims) * 0.1, dtype=fX)


def main():
    H, B = 10, 32
    model = PolicyNetwork(H, B)

    print([parameter.get_value().shape for parameter in model.parameters])

    action = model.actionFunction([1] * (H + B))
    print(action.shape, 1 * (action > 0.5))


if __name__ == '__main__':
    main()
