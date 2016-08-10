#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
import os
import json
import numpy as np

from config import Config

__author__ = 'fyabc'


def getTokensMap():
    tokens2index = json.load(open(
        os.path.join(Config['preprocessDir'], Config['preprocessFiles']['tokens2index']), 'r'))
    index2tokens = json.load(open(
        os.path.join(Config['preprocessDir'], Config['preprocessFiles']['index2tokens']), 'r'))
    return tokens2index, index2tokens


def token2vector(tokens2index, token):
    result = np.zeros(shape=(len(tokens2index), 1), dtype='float32')
    result[tokens2index[token]] = 1.0
    return result
