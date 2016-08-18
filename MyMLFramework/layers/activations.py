#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
import theano.tensor as T

from .layer import NNLayer

__author__ = 'fyabc'


def softplus(x):
    return T.nnet.softplus(x)


def relu(x):
    return T.nnet.relu(x)


def tanh(x):
    return T.tanh(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)


def linear(x):
    return x


class Activation(NNLayer):
    def __init__(self, activationType, **kwargs):
        self.activationType = activationType
        self.kwargs = kwargs
        super(Activation, self).__init__(None)

    def setOutputShape(self):
        self.outputShape = self.inputShape

    def makeFunction(self):
        try:
            self.function = globals()[self.activationType]
        except KeyError:
            raise Exception('Cannot find such activation type.')
