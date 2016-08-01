#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
import random

__author__ = 'fyabc'

'''
    A simple robot sample to learn MDP.
    [Page](http://suanfazu.com/t/topic/13571)

    1 2 3 4 5
    * * * * *
    -   +   -
    6   7   8

    A = {'n', 'e', 's', 'w'}
'''


class MDP:
    STATE_NUM = 8

    def __init__(self, gamma=0.5):
        self.states = [i for i in range(1, 1 + self.STATE_NUM)]
        self.terminalStates = {6, 7, 8}

        self.actions = ('n', 'e', 's', 'w')

        self.rewards = {
            (1, 's'): -1.0,
            (3, 's'): 1.0,
            (5, 's'): -1.0,
        }

        self.t = {
            (1, 's'): 6,
            (1, 'e'): 2,
            (2, 'w'): 1,
            (2, 'e'): 3,
            (3, 's'): 7,
            (3, 'w'): 2,
            (3, 'e'): 4,
            (4, 'w'): 3,
            (4, 'e'): 5,
            (5, 's'): 8,
            (5, 'w'): 4,
        }

        self.gamma = gamma

    def transform(self, state, action):
        """
        :return: is_terminal, state, reward
        """
        if state in self.terminalStates:
            return True, state, 0.0

        key = (state, action)

        nextState = self.t.get(key, state)
        isTerminal = nextState in self.terminalStates
        reward = self.rewards.get(key, 0.0)

        return isTerminal, nextState, reward

    def randomAction(self):
        return random.choice(self.actions)

    def randomWalk(self, sampleNumber=1):
        stateSamples, actionSamples, rewardSamples = [], [], []

        for _ in range(sampleNumber):
            statesTmp, actionsTmp, rewardsTmp = [], [], []

            state = random.choice(self.states)
            isTerminal = False

            while not isTerminal:
                action = self.randomAction()
                isTerminal, nextState, reward = self.transform(state, action)

                statesTmp.append(state)
                actionsTmp.append(action)
                rewardsTmp.append(reward)

                state = nextState

            stateSamples.append(statesTmp)
            actionSamples.append(actionsTmp)
            rewardSamples.append(rewardsTmp)

        return stateSamples, actionSamples, rewardSamples


if __name__ == '__main__':
    print('In MDP class')