#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
from MDP import MDP
import version23

__author__ = 'fyabc'

# random.seed(0)


def getRandomPolicyValue():
    values = [0.0 for _ in range(10)]
    num = 1000000
    echoEpoch = 10000

    mdp = MDP()

    for k in range(1, num):
        for initState in range(1, 6):
            state = initState
            isTerminal = False
            gamma = 1.0
            value = 0.0

            while not isTerminal:
                action = mdp.randomAction()
                isTerminal, state, reward = mdp.transform(state, action)
                value += gamma * reward
                gamma *= mdp.gamma

            values[initState] += value

        if k % echoEpoch == 0:
            print('k = %d, Average values of state 1-5 are:\n' % k,
                  [value / k for value in values[1:6]])

    for i in range(len(values)):
        values[i] /= num

    return values


def test():
    values = getRandomPolicyValue()
    print('Average values of state 1-5 are:\n', values[1:6])


if __name__ == '__main__':
    test()
