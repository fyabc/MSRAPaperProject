#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
import theano.tensor as T

__author__ = 'fyabc'


class Regularizer(object):
    def __init__(self):
        self.function = None

    def __call__(self, parameters):
        return self.function(parameters)


class L2(Regularizer):
    def __init__(self, lambda_=0.1):
        self.lambda_ = lambda_
        super(L2, self).__init__()
        self.function = lambda parameters: self.lambda_ * T.sum([T.sum(parameter ** 2) for parameter in parameters])
