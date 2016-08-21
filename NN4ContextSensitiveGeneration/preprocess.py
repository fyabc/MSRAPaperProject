#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
import os
import json
import time
from config import Config, ParamConfig

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


def getCharIterator(inputList, inputType='file'):
    if inputType == 'file':
        for fileName in inputList:
            with open(fileName, 'r') as f:
                for line in f:
                    for char in line:
                        yield char
    elif inputType == 'str':
        for string in inputList:
            for char in string:
                yield char


def getTokensMap(iterTokens, defaultTokens=('<SOS>', '<EOS>', '<UNK>')):
    """

    :param iterTokens: tokens iterator.
    :param defaultTokens: default tokens to be add.
        <SOS>:  Start of sentence
        <EOS>:  End of sentence
        <UNK>:  Unknown
    :return:
    """

    tokens2Index = {}
    index2Tokens = {}

    class _TokenAdder(object):
        def __init__(self):
            self.currentIndex = 0

        def add(self, token_):
            if token_ not in tokens2Index:
                tokens2Index[token_] = self.currentIndex
                index2Tokens[self.currentIndex] = token_
                self.currentIndex += 1

    _adder = _TokenAdder()

    for token in defaultTokens:
        _adder.add(token)

    for token in iterTokens:
        _adder.add(token)

    return tokens2Index, index2Tokens


def getFileNameList():
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

    return fileNameList


def parseTwitterData(level=ParamConfig['level']):
    """
    Parse Twitter data, and dump it into JSON files.
    :param level: the level of tokenizing. 'Char' or 'Tokens'.
    :return: the number of all different tokens.
    """

    fileNameList = getFileNameList()

    tokens2Index, index2Tokens = getTokensMap(eval('get%sIterator(fileNameList)' % level))

    with open(os.path.join(Config['preprocessDir'], Config['preprocessFiles']['tokens2index']), 'w') as t2i:
        json.dump(tokens2Index, t2i)
    with open(os.path.join(Config['preprocessDir'], Config['preprocessFiles']['index2tokens']), 'w') as i2t:
        json.dump(index2Tokens, i2t)

    return len(tokens2Index)


def readTokensMapFile():
    with open(os.path.join(Config['preprocessDir'], Config['preprocessFiles']['tokens2index']), 'r') as t2i:
        tokens2Index = json.load(t2i)
    with open(os.path.join(Config['preprocessDir'], Config['preprocessFiles']['index2tokens']), 'r') as i2t:
        index2Tokens = json.load(i2t)

    return tokens2Index, index2Tokens


def test():
    timeBefore = time.time()
    tokenNumber = parseTwitterData()
    timeAfter = time.time()
    print('Tokens number = %d, time passed = %.3fs' % (tokenNumber, timeAfter - timeBefore))


if __name__ == '__main__':
    test()
