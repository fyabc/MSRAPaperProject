#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
from operator import gt

import version23

__author__ = 'fyabc'


def argmax(seq, cmpFunc=gt):
    if len(seq) == 0:
        return None

    maxValue = seq[0]
    maxIndex = 0

    for i, elem in seq:
        if cmpFunc(elem, maxValue):
            maxValue = elem
            maxIndex = i

    return maxIndex


bestQFunc = {}


def getBestQFunc():
    if len(bestQFunc) > 0:
        return

    with open('bestQFunc.txt', 'r') as f:
        for line in f:
            words = line.split()
            if len(words) >= 3:
                bestQFunc[int(words[0]), words[1]] = float(words[2])


def getSquareError(qFunc):
    return sum(map(lambda key: (qFunc[key] - bestQFunc[key]) ** 2, qFunc), 0)


def getSquareErrorPolicy(policy):
    result = 0.0

    for key in bestQFunc:
        state, action = key
        if state in policy.mdp.terminalStates:
            continue

        error = policy.qFunc(policy.mdp.getFeature(state), action) - bestQFunc[key]
        result += error ** 2

    return result
