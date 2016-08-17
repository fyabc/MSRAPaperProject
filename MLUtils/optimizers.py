#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from abc import ABCMeta, abstractmethod

import theano.tensor as T
from theano import function

__author__ = 'fyabc'


class Optimizer(object):
    __metaclass__ = ABCMeta

    def __init__(self, cost, parameters):
        self.cost = cost

        if not isinstance(parameters, (list, tuple)):
            parameters = (parameters,)
        self.parameters = parameters
        self.step = None

    def getGParameters(self):
        return [T.grad(self.cost, parameter) for parameter in self.parameters]

    @abstractmethod
    def build(self, inputs):
        pass

    def train(self, data, batchSize=1, epsilon=1e-6):
        """
        optimize the cost on data.
        must be called after `build`.

        :param data: input data.
            should be a sequence of training records, will be split into mini-batches.
            number of records must be the first dimension of data.
            each record should be a sequence that contains all input data in the same order of self.variables.

        :param batchSize: size of one mini-batch.
        :param epsilon: the threshold that end the iteration.
        """

        stepNum = 0
        running = True

        while running:
            batchNum = data.shape[0] // batchSize + (len(data) % batchSize != 0)
            for i in range(batchNum):
                cost = self.step(*[data[i * batchSize: (i + 1) * batchSize, j] for j in range(data.shape[1])])

                if abs(cost) < epsilon:
                    running = False
                    break
                stepNum += 1


class SGD(Optimizer):
    def __init__(self, cost, inputs, parameters, learningRate=0.01):
        super(SGD, self).__init__(cost, parameters)
        self.learningRate = learningRate

        self.build(inputs)

    def build(self, inputs):
        gVariables = self.getGParameters()
        updates = [(parameter, parameter - self.learningRate * gVariable)
                   for parameter, gVariable in zip(self.parameters, gVariables)]

        self.step = function(
            inputs=inputs,
            outputs=self.cost,
            updates=updates,
        )


class Momentum(Optimizer):
    def __init__(self, cost, inputs, parameters, learningRate, alpha):
        super(Momentum, self).__init__(cost, parameters)
        self.learningRate = learningRate
        self.alpha = alpha

        self.build(inputs)

    def build(self, inputs):
        gVariables = self.getGParameters()


# Some simple optimizers for currently use.


def simpleSGD(cost, inputs, variables, learningRate=0.01):
    gVariables = [T.grad(cost, variable) for variable in variables]
    updates = [(variable, variable - learningRate * gVariable) for variable, gVariable in zip(variables, gVariables)]

    return function(
        inputs=inputs,
        outputs=cost,
        updates=updates,
    )


def testSGD():
    import math
    import numpy as np
    from theano import shared

    targetA = 3.0
    target = lambda x, a=targetA: math.sin(a * x) + a + 1.0

    a = shared(-10.0, name='a')
    x = T.matrix('x', dtype=a.dtype)
    y = T.matrix('y', dtype=a.dtype)
    fx = T.sin(a * x) + a + 1.0
    cost = T.sum((fx - y) ** 2)
    optimizer = SGD(cost, [x, y], [a])

    data = np.asmatrix([[_, target(_)] for _ in np.linspace(-10.0, 10.0, 400)], dtype=a.dtype)
    optimizer.train(data, batchSize=3)

    print(a.get_value())


def test():
    testSGD()


if __name__ == '__main__':
    test()
