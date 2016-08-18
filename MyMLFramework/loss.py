#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
import theano.tensor as T

__author__ = 'fyabc'


class Loss(object):
    def __init__(self):
        self.function = None

    def __call__(self, output, target):
        return self.function(output, target)


class MSE(Loss):
    """
    Minimum Square Error.
    """
    def __init__(self):
        super(MSE, self).__init__()
        self.function = lambda output, y: T.sum((output - y) ** 2)
