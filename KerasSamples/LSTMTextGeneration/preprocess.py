#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import os
import numpy as np
from config import Config, ParamConfig

__author__ = 'fyabc'


def _joinPath(*keys):
    return os.path.join(*[Config[key] for key in keys])


def readFile():
    return open(_joinPath('dataDir', 'dataFiles')).read().lower()


def getChars(text):
    return sorted(set(text))


def getCharMap(chars):
    return {c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(chars)}


def getSentences(text, maxLen=ParamConfig['maxLen'], step=ParamConfig['step']):
    sentences = []
    nextChars = []

    for i in range(0, len(text) - maxLen, step):
        sentences.append(text[i: i + maxLen])
        nextChars.append(text[i + maxLen])

    return sentences, nextChars


def sentence2vector(sentences, nextChars, chars, char2index, maxLen=ParamConfig['maxLen']):
    X = np.zeros((len(sentences), maxLen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char2index[char]] = 1
        y[i, char2index[nextChars[i]]] = 1

    return X, y


def test():
    text = readFile()
    print(text)
    print(getChars(text))


if __name__ == '__main__':
    test()
