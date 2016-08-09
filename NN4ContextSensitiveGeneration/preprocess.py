#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
import os
import json
import time
from config import Config

__author__ = 'fyabc'


"""
This file do the preprocessing of Twitter data.
"""


def getTokensIterator(inputList, inputType='file'):
    """
    :param inputList: list of input.
        if inputType is 'file', it is a list of file names.
        if inputType is 'str', it is a list of strings.

    :param inputType: type of input.
        now support two types: 'file' and 'str'.
    :return: an iterator which generate tokens from all inputs.
    """

    if inputType == 'file':
        for fileName in inputList:
            with open(fileName, 'r') as f:
                for line in f:
                    words = line.split()
                    for word in words:
                        yield word
    elif inputType == 'str':
        for string in inputList:
            words = string.split()
            for word in words:
                yield word


def getTokensMap(iterTokens):
    tokens2Index = {}
    index2Tokens = {}

    currentIndex = 0
    for token in iterTokens:
        if token not in tokens2Index:
            tokens2Index[token] = currentIndex
            index2Tokens[currentIndex] = token
            currentIndex += 1

    return tokens2Index, index2Tokens


def parseTwitterData():
    fileNameList = [
        Config['dataFiles']['train'][0],
        Config['dataFiles']['train'][1],
        Config['dataFiles']['valid'][0],
        Config['dataFiles']['valid'][1],
        Config['dataFiles']['test'][0],
        Config['dataFiles']['test'][1],
    ]

    for i in range(len(fileNameList)):
        fileNameList[i] = os.path.join(Config['dataDir'], fileNameList[i])

    tokens2Index, index2Tokens = getTokensMap(getTokensIterator(fileNameList))

    with open(os.path.join(Config['preprocessDir'], Config['preprocessFiles']['tokens2index']), 'w') as t2i:
        json.dump(tokens2Index, t2i)
    with open(os.path.join(Config['preprocessDir'], Config['preprocessFiles']['index2tokens']), 'w') as i2t:
        json.dump(index2Tokens, i2t)

    return len(tokens2Index)


def test():
    timeBefore = time.time()
    tokenNumber = parseTwitterData()
    timeAfter = time.time()
    print('Tokens number = %d, time passed = %.3fs' % (tokenNumber, timeAfter - timeBefore))


if __name__ == '__main__':
    test()
