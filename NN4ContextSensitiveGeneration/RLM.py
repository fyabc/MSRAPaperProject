#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
import numpy as np
import theano.tensor as T
from theano import shared, function, config, scan

from config import Config, ParamConfig
from preprocess import *

__author__ = 'fyabc'

fX = config.floatX


def toFX(value):
    return eval('np.%s(value)' % fX)


class RLM(object):
    r"""
    RNN Language Model.

    Hyper-parameters and parameters:
        K                           size of vocabulary
        T                           length of sentence
        H                           number of hidden neurons

        x = {x_1 ~ x_T}             word sequence of the sentence x
        x_t (in R^{K x 1})          input of RNN @ time t, one-hot vector
        \hat{y}_t (in R^{K x 1})    output of RNN @ time t, the probability of each word
        y_t (in R^{K x 1})          label of RNN @ time t, one-hot vector

        E_t                         loss function @ time t, defined as cross entropy (see below)
        E                           loss function of the whole sentence, sum of E_t over all t in 1 ~ T.
                                    (we think about SGD, so we just get the loss of one sentence)

        s_t (in R^{H x 1})          input of hidden layer @ time t
        h_t (in R^{H x 1})          output of hidden layer @ time t
        z_t (in R^{K x 1})          pooling input of output layer @ time t
        f_1, f_2                    activation function of hidden layer and output layer (tanh, sigmoid, etc)
                                    f_2 should be sigmoid to output values between 0 and 1?

        r_t = \hat{y}_t - y_t       residual vector

        W (in R^{H x K})            weights of input -> hidden
        U (in R^{H x H})            weights of hidden -> hidden'
        V (in R^{K x H})            weights of hidden -> output

    Equations:
        s_t = U * h_{t-1} + W * x_t     [hidden]
        h_t = f_1(s_t)                  [hidden a]

        z_t = V * h_t                   [output]
        \hat{y}_t = f_2(z_t)            [output a]

        E_t = -y_t' * log(\hat{y}_t)
        E = \sigma{t = 1 ~ T}(E_t)

    Target:
        get dE / dU, dE / dV, dE / dW.
    """

    def __init__(self, H=ParamConfig['H']):
        # Shapes:
        # x: (T, K)
        # y: (T, K)
        #
        # W: (H, K)
        # U: (H, H)
        # V: (K, H)
        #
        # h: (T, H)
        # hat_y: (T, K)

        # Tokens map files
        self.tokens2Index, self.index2Tokens = readTokensMapFile()
        self.level = ParamConfig['level']

        # Some hyper-parameters
        self.K = len(self.tokens2Index)
        self.H = H
        self.learningRate = shared(toFX(0.05))

        # Parameters to be trained
        self.W = shared(name='W', value=self.initNorm(self.H, self.K))
        self.U = shared(name='U', value=self.initNorm(self.H, self.H))
        self.V = shared(name='V', value=self.initNorm(self.K, self.H))
        self.parameters = [self.W, self.U, self.V]

        # Inputs
        self.x = T.matrix(name='x', dtype=fX)
        self.y = T.matrix(name='y', dtype=fX)

        h_init = T.vector(name='h_init', dtype=fX)

        def _step(x_t, h_prev):
            s_t = T.dot(self.U, h_prev) + T.dot(self.W, x_t)
            h_t = T.tanh(s_t)

            z_t = T.dot(self.V, h_t)
            # hat_y_t = T.nnet.softmax(z_t)
            # hat_y_t = hat_y_t.reshape((hat_y_t.shape[1],))
            hat_y_t = T.nnet.sigmoid(z_t)

            return h_t, hat_y_t

        (self.h, self.hat_y), updates = scan(
            fn=_step,
            sequences=self.x,
            outputs_info=[h_init, None],    # None must be added because we don't need hat_y_t as step input.
        )

        # The function to get output
        self.outputFunction = function(
            inputs=[self.x],
            outputs=[self.h, self.hat_y],
            updates=updates,
            givens=((h_init, self.initNorm(self.H)),)
        )

        # The loss (cross entropy)
        self.E = T.sum(-self.y * T.log(self.hat_y))\
            # + 0.1 * T.sum([T.sum(parameter ** 2) for parameter in self.parameters])

        # SGD optimizer
        updates = [(parameter, parameter - self.learningRate * T.grad(self.E, parameter))
                   for parameter in self.parameters]

        self.sgd = function(
            inputs=[self.x, self.y],
            outputs=self.E,
            updates=updates,
            givens=((h_init, self.initNorm(self.H)),)
        )

    @staticmethod
    def initNorm(*dims):
        return np.asarray(np.random.randn(*dims) * 0.1, dtype=fX)

    def tokens2Matrix(self, tokens):
        result = np.zeros((len(tokens), self.K), dtype=fX)
        for i, token in enumerate(tokens):
            result[i][self.tokens2Index[token]] = 1
        return result

    def matrix2Tokens(self, hat_y):
        iterTokens = (self.index2Tokens[str(index)] for index in np.argmax(hat_y, axis=1))
        if self.level == 'Char':
            return ''.join(iterTokens)
        elif self.level == 'Tokens':
            return ' '.join(iterTokens)

    @staticmethod
    def accuracy(y, hat_y):
        return sum((i1 == i2 for i1, i2 in zip(np.argmax(y, axis=1), np.argmax(hat_y, axis=1))), 0.0) / len(hat_y)

    def iterSentences(self, fileName):
        with open(fileName, 'r') as f:
            for line in f:
                if self.level == 'Char':
                    yield line
                elif self.level == 'Tokens':
                    yield line.split()


def test():
    model = RLM()

    print('The vocabulary size:', model.K)
    print('The hidden size:', model.H)

    postTrain, resTrain, postValid, trainValid, postTest, resTest = getFileNameList()

    losses = []

    for i, (post, res) in enumerate(zip(model.iterSentences(postTrain), model.iterSentences(resTrain))):
        x = model.tokens2Matrix(post)
        y = model.tokens2Matrix(res)

        # filter input data
        minLen = min(len(x), len(y))
        x, y = x[:minLen], y[:minLen]

        loss = model.sgd(x, y)
        losses.append(loss)

        if i % 10000 == 0:
            model.learningRate.set_value(model.learningRate.get_value() * toFX(0.6))

        if i % 1000 == 0:
            h, hat_y = model.outputFunction(x)
            post = post[:-1] if post[-1] == '\n' else post
            res = res[:-1] if res[-1] == '\n' else res
            genRes = model.matrix2Tokens(hat_y)
            print('Iter         :', i)
            print('Learning Rate:', model.learningRate.get_value())
            print('Loss         :', loss)
            print('Post         :', post)
            print('Res          :', res)
            print('Gen res      :', genRes)
            print('Accuracy     :', model.accuracy(y, hat_y))
            print()

    import matplotlib.pyplot as plt
    plt.plot(list(xrange(i + 1)), losses)
    plt.show()


if __name__ == '__main__':
    test()
