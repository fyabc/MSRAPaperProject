#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import unicode_literals, print_function

import cPickle as pkl
import json

import theano.tensor as T
from theano import config, function

from loss import MSE
from regularizers import L2
from optimizers import SGD

__author__ = 'fyabc'


class SimpleModel(object):
    """
    A simple neural network model.
    """

    def __init__(self):
        self.parameters = []
        self.input = None
        self.output = None
        self.loss = None

        self.objectiveFunction = None

    def setInput(self, ndim, dtype=config.floatX):
        self.input = T.tensor(name='input', dtype=dtype, broadcastable=(False,) * ndim)
        self.output = self.input

    def addRaw(self, operation, parameters):
        """
        Add a layer directly by add its operation and parameters.
        :param operation: a callable, input from previous layer, then get output of this layer.
        :param parameters: parameters to be optimized, should be Theano's shared variables.
        """

        self.output = operation(self.output)
        for parameter in parameters:
            self.parameters.append(parameter)

    def compile(self, optimizer=None, loss=None):
        if loss is None:
            loss = MSE()

        target = self.output.clone()
        self.loss = loss(self.output, target)

        self.objectiveFunction = function(
            inputs=[self.input, target],
            outputs=self.loss
        )


class Model(object):
    """
    A more structured model.
    """

    def __init__(self):
        self.parameters = []

        self.input = None
        self.inputShape = None
        self.output = None
        self.outputShape = None

        self._resultFunction = None
        self.objectiveFunction = None

        self.layers = []

    def _setInput(self, dtype=config.floatX):
        if self.input is not None or self.inputShape is None:
            return

        self.input = T.tensor(name='input', dtype=dtype, broadcastable=(False,) * len(self.inputShape))
        self.output = self.input

    def add(self, layer):
        if layer.inputShape is None:
            if self.inputShape is None:
                raise Exception('The first layer in this model must have an inputShape.')
            else:
                layer.inputShape = self.outputShape
                if layer.outputShape is None:
                    layer.setOutputShape()
                layer.makeFunction()
        else:
            if self.outputShape is not None:
                assert layer.inputShape == self.outputShape,\
                    'The inputShape of layer does not match the current outputShape of model.'
            if self.inputShape is None:
                self.inputShape = layer.inputShape

        self._setInput()

        self.layers.append(layer)
        self.output = layer.function(self.output)
        for parameter in layer.parameters:
            self.parameters.append(parameter)

        self.outputShape = layer.outputShape

    @property
    def resultFunction(self):
        if self._resultFunction is None:
            self._resultFunction = function(
                inputs=self.input,
                outputs=self.output,
            )
        return self._resultFunction

    def compile(self, optimizer=None, loss=None, regularize=None):
        if optimizer is None:
            optimizer = SGD()

        if loss is None:
            loss = MSE()

        if regularize is None:
            # regularize = L2()
            regularize = lambda parameters: 0

        target = self.output.clone()
        lossValue = loss(self.output, target) + regularize(self.parameters)

        self.objectiveFunction = optimizer.getFunction(lossValue, self.input, target, self.parameters)

    def saveParameters(self, file_, type_='pkl'):
        values = [parameter.get_value() for parameter in self.parameters]
        if type_ == 'pkl':
            pkl.dump(values, file_)
        elif type_ == 'json':
            json.dump(values, file_)
        else:
            raise ValueError('Cannot save parameters in type {}'.format(type_))

    def loadParameters(self, file_, type_='pkl'):
        pass
