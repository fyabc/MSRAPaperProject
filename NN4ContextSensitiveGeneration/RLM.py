#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from keras.models import Sequential
from keras.layers import SimpleRNN

__author__ = 'fyabc'

config = {
    'K': 800,
    'H': 200,
}


def RLM():
    model = Sequential()
    model.add(SimpleRNN(config['H'], input_shape=(config['K'],)))


def test():
    pass


if __name__ == '__main__':
    test()
