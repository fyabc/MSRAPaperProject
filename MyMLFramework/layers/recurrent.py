#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from .layer import NNLayer

__author__ = 'fyabc'


class Recurrent(NNLayer):
    def __init__(self, outputShape, inputShape=None):
        super(Recurrent, self).__init__(outputShape, inputShape)


class SimpleRNN(Recurrent):
    pass


class GRU(Recurrent):
    pass


class LSTM(Recurrent):
    pass
