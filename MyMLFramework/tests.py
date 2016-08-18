#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import unicode_literals, print_function

import numpy as np
import theano.tensor as T
from theano import shared, config

from MyMLFramework.model import SimpleModel, Model
from layers.layer import Dense
from layers.activations import Activation

__author__ = 'fyabc'

fX = config.floatX


def testSimpleModel():
    # config.optimizer = 'None'
    # config.exception_verbosity = 'high'

    # def initNorm(*args):
    #     return np.asarray(np.random.randn(*args) * 0.1, dtype=fX)

    model = SimpleModel()
    model.setInput(2)

    W1 = shared(value=np.asarray([
        [1, 2, 3],
        [4, 0, 1],
    ], dtype=fX), name='W1')
    b1 = shared(value=np.asarray([
        1, 2, 4,
    ], dtype=fX), name='b1')

    def layer1(x):
        return T.nnet.relu(T.dot(x, W1) + b1)

    model.addRaw(layer1, [W1, b1])

    model.compile()

    result = model.objectiveFunction([[6, 1], [7, 2], [8, 3]], [[7, 6, 2], [1, 2, 3], [4, 0, 5]])

    print(result)


def testModel():
    model = Model()
    model.add(Dense(
        outputShape=(3, 3,),
        inputShape=(3, 2,)
    ))
    model.add(Activation('sigmoid'))

    model.compile()

    result = model.objectiveFunction([[6, 1], [7, 2], [8, 3]], [[7, 6, 2], [1, 2, 3], [4, 0, 5]])

    print(result)


def test():
    # testSimpleModel()
    testModel()


if __name__ == '__main__':
    test()
