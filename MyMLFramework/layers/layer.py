#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
import numpy as np
import theano.tensor as T
from theano import shared

__author__ = 'fyabc'


class NNLayer(object):
    def __init__(self, outputShape, inputShape=None):
        self.outputShape = outputShape
        self.inputShape = inputShape
        self.function = None
        self.parameters = []

        if self.inputShape is not None:
            self.makeFunction()

    def setOutputShape(self):
        """set outputShape of this layer by its inputShape and other parameters."""
        pass

    def makeFunction(self):
        """
        make the function of this layer.
        [NOTE]: the input and the output must be given.
        """
        pass


def _initNorm(*args):
    from theano import config
    return np.asarray(np.random.randn(*args) * 0.1, dtype=config.floatX)


class Dense(NNLayer):
    """
    A dense layer (full connection)
    output = input .* W + b
    """

    def __init__(self, outputShape, inputShape=None, initType='norm'):
        self.initType = initType
        super(Dense, self).__init__(outputShape, inputShape)

    def makeFunction(self):
        self.parameters = [
            shared(
                value=_initNorm(self.inputShape[-1], self.outputShape[-1]),
                name='W'
            ),
            shared(
                value=_initNorm(self.outputShape[-1]),
                name='b'
            )
        ]
        self.function = lambda x: T.dot(x, self.parameters[0]) + self.parameters[1]
