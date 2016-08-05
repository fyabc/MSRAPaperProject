#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import random
import numpy as np

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


class MDP(object):
    STATE_NUM = 8

    def __init__(self, gamma=0.8):
        self.states = [i for i in range(1, 1 + self.STATE_NUM)]
        self.terminalStates = {6, 7, 8}

        self.actions = ('n', 'e', 's', 'w')

        self.rewards = {
            (1, 's'): -1.0,
            (3, 's'): 1.0,
            (5, 's'): -1.0,
        }

        self.transTable = {
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

    @property
    def actionSize(self):
        return len(self.actions)

    @property
    def stateSize(self):
        return self.STATE_NUM

    @property
    def terminalStateSize(self):
        return len(self.terminalStates)

    def transform(self, state, action):
        """
        :return: is_terminal, state, reward
        """
        if state in self.terminalStates:
            return True, state, 0.0

        key = (state, action)

        nextState = self.transTable.get(key, state)
        isTerminal = nextState in self.terminalStates
        reward = self.rewards.get(key, 0.0)

        return isTerminal, nextState, reward

    def randomAction(self):
        return random.choice(self.actions)

    def randomWalk(self):
        """
        :return: states, actions, rewards
        """
        states, actions, rewards = [], [], []

        state = random.choice(self.states)
        isTerminal = False

        while not isTerminal:
            action = self.randomAction()
            isTerminal, nextState, reward = self.transform(state, action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = nextState

        return states, actions, rewards

    def randomWalkSamples(self, sampleNumber=1):
        """
        :return: list of results of randomWalk()
        """
        stateSamples, actionSamples, rewardSamples = [], [], []

        for _ in range(sampleNumber):
            statesTmp, actionsTmp, rewardsTmp = self.randomWalk()

            stateSamples.append(statesTmp)
            actionSamples.append(actionsTmp)
            rewardSamples.append(rewardsTmp)

        return stateSamples, actionSamples, rewardSamples


class FeatureMDP(MDP):
    def __init__(self, gamma=0.8, feature='wall'):
        super(FeatureMDP, self).__init__(gamma)

        self.featureType = feature
        self.features = {}
        if feature == 'wall':
            self.features[1] = np.array([1, 0, 0, 1])
            self.features[2] = np.array([1, 0, 1, 0])
            self.features[3] = np.array([1, 0, 0, 0])
            self.features[4] = np.array([1, 0, 1, 0])
            self.features[5] = np.array([1, 1, 0, 0])
            self.features[6] = np.array([0, 1, 1, 1])
            self.features[7] = np.array([0, 1, 1, 1])
            self.features[8] = np.array([0, 1, 1, 1])
        elif feature == 'identity':
            for i in range(1, self.STATE_NUM + 1):
                self.features[i] = np.array([0 for _ in range(self.STATE_NUM)])
                self.features[i][i - 1] = 1
        else:
            raise KeyError('Unknown feature type')

    @property
    def sFeatureSize(self):
        return len(self.features[self.states[0]])

    def getFeature(self, state):
        """can be implemented by dict lookup or calculate."""
        return self.features[state]

    def transform(self, state, action):
        isTerminal, nextState, reward = super(FeatureMDP, self).transform(state, action)
        return isTerminal, nextState, reward, self.getFeature(nextState)


def test():
    mdp = MDP()

    states, actions, rewards = mdp.randomWalk()

    print('A random walk:')
    for i in range(len(states)):
        print('I am in state %d' % (states[i], ))
        print('Action: %s' % (actions[i], ))
    print('I am in state %d' % (mdp.transTable.get((states[-1], actions[-1]), states[-1]), ))
    print('The last result is %2.1f' % (rewards[-1], ))


if __name__ == '__main__':
    test()
