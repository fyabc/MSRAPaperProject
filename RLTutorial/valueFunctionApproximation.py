#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import random

from MDP import MDP
from utils import getSquareErrorPolicy, getBestQFunc, bestQFunc
import version23


__author__ = 'fyabc'


r"""
When the state space is very large (such as Go, etc.), we cannot save or compute v(s) and q(s, a).
So we use feature extraction:
    For a state s, use its feature \hat{s} instead of s to compute the v and q functions.
    It is called value function approximation.

In this toy robot model, we apply 2 features:
    1. feature vector (wall feature as example)
        \hat{s} = [@n, @e, @s, @w], @x = 1 means there is a wall in the direction x.
        Eg. \hat{s_1} = [1, 0, 0, 1]

        the model-free algorithms need to compute q(s, a),
        so the feature vector need contain features of s and a.

        let the feature vector f(\hat{s}, a) be:
            for all actions a' in A:
                if a' == a then the part of f is the feature vector of s
                else all zero.
        Eg. f(\hat{s_1}, 'n') = [1 0 0 1, 0 0 0 0, 0 0 0 0, 0 0 0 0]

    2. get other functions
        q(\hat{s}, a) = f(\hat{s}, a)' * w
        w is parameters.

        objective function is:
        J(w) = \sigma{s in S, a in A}( (q(\hat{s}, a) - q_best(s, a)) ** 2 )

        To minimize J:
        grad_{J(w)} = \sigma{s in S, a in A}( f(\hat{s}, a) * (q(\hat{s}, a) - q_best(s, a)) )

        But S is very large, so we use SGD method:
        grad_{J(w)}(\hat{s}, a) = f(\hat{s}, a) * (q(\hat{s}, a) - q_best(s, a))

        \delta_w = -\alpha * grad_{J(w)}(\hat{s}, a)

    3. reinforcement learning algorithms
        How to get q_best ? use model-free learning algorithms.
            qFunc = g_t                                     (MC Control)
            qFunc = r + \gamma * q(\hat{s'}, a')            (SARSA)
            qFunc = r + max_{a'}(\gamma * q(\hat{s'}, a'))  (Q-Learning)
"""


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
    def stateFeatureSize(self):
        return len(self.features[self.states[0]])

    def getFeature(self, state):
        """can be implemented by dict lookup or calculate."""
        return self.features[state]

    def transform(self, state, action):
        isTerminal, nextState, reward = super(FeatureMDP, self).transform(state, action)
        return isTerminal, nextState, reward, self.getFeature(nextState)


class Policy(object):
    def __init__(self, featureMDP):
        self.mdp = featureMDP

        # parameters is w
        self.parameters = np.zeros((self.mdp.actionSize * self.mdp.stateFeatureSize,), dtype='float64')

    def getActionFeatureVector(self, stateFeature, action):
        result = np.zeros_like(self.parameters, dtype='int32')

        sfSize = self.mdp.stateFeatureSize
        for index, a in enumerate(self.mdp.actions):
            if a == action:
                result[index * sfSize:(index + 1) * sfSize] = stateFeature
                break

        return result

    def qFunc(self, stateFeature, action):
        return np.dot(self.getActionFeatureVector(stateFeature, action), self.parameters)

    def epsilonGreedy(self, stateFeature, epsilon):
        # argmax_a(q(\hat{s}, a))
        maxAIndex = 0
        maxQ = self.qFunc(stateFeature, self.mdp.actions[0])

        for i in range(self.mdp.actionSize):
            a = self.mdp.actions[i]
            q = self.qFunc(stateFeature, a)
            if maxQ < q:
                maxQ = q
                maxAIndex = i

        temp = epsilon / self.mdp.actionSize

        # random choose
        choice = random.random()

        # get action chosen
        sumProb = 0.0
        for i in range(self.mdp.actionSize):
            if i == maxAIndex:
                sumProb += 1 - epsilon + temp
            else:
                sumProb += temp
            if sumProb >= choice:
                return self.mdp.actions[i]
        return self.mdp.actions[-1]


def updatePolicy(policy, stateFeature, action, bestQValue, alpha):
    """
    alpha: learning rate.
    """
    qValue = policy.qFunc(stateFeature, action)
    error = qValue - bestQValue

    feature = policy.getActionFeatureVector(stateFeature, action)
    policy.parameters -= alpha * error * feature


def featureMCControl(mdp, epsilon, alpha, iterNum, maxWalkLen=100, echoSE=False):
    """
    qFunc = g_t
    """

    InitParameter = 0.1

    if echoSE:
        squareErrors = []

    policy = Policy(mdp)

    for i in range(len(policy.parameters)):
        policy.parameters[i] = InitParameter

    for _ in range(iterNum):
        if echoSE:
            squareErrors.append(getSquareErrorPolicy(policy))

        states, sFeatures, actions, rewards = [], [], [], []

        state = random.choice(mdp.states)
        sFeature = mdp.getFeature(state)
        isTerminal = False

        count = 0
        while not isTerminal and count < maxWalkLen:
            action = policy.epsilonGreedy(sFeature, epsilon)
            isTerminal, nextState, reward, nextSFeature = mdp.transform(state, action)

            states.append(state)
            sFeatures.append(sFeature)
            rewards.append(reward)
            actions.append(action)

            state = nextState
            sFeature = nextSFeature
            count += 1

        g = 0.0
        for i in range(len(states) - 1, -1, -1):
            g *= mdp.gamma
            g += rewards[i]

        for i in range(len(states)):
            updatePolicy(policy, sFeatures[i], actions[i], g, alpha)

            g -= rewards[i]
            g /= mdp.gamma

    if echoSE:
        return policy, squareErrors
    else:
        return policy


def featureSARSA(mdp, epsilon, alpha, iterNum, maxWalkLen=100, echoSE=False):
    """
    qFunc = r + \gamma * q(\hat{s'}, a')
    """

    InitParameter = 0.1

    if echoSE:
        squareErrors = []

    policy = Policy(mdp)

    for i in range(len(policy.parameters)):
        policy.parameters[i] = InitParameter

    for _ in range(iterNum):
        if echoSE:
            squareErrors.append(getSquareErrorPolicy(policy))

        state = random.choice(mdp.states)
        sFeature = mdp.getFeature(state)
        action = random.choice(mdp.actions)
        isTerminal = False

        count = 0
        while not isTerminal and count < maxWalkLen:
            isTerminal, nextState, reward, nextSFeature = mdp.transform(state, action)
            nextAction = policy.epsilonGreedy(nextSFeature, epsilon)

            updatePolicy(policy, sFeature, action, reward + mdp.gamma * policy.qFunc(nextSFeature, nextAction), alpha)

            state = nextState
            sFeature = nextSFeature
            action = nextAction
            count += 1

    if echoSE:
        return policy, squareErrors
    else:
        return policy


def featureQLearning(mdp, epsilon, alpha, iterNum, maxWalkLen=100, echoSE=False):
    """
    qFunc = r + max_{a'}(\gamma * q(\hat{s'}, a'))
    """

    InitParameter = 0.1

    if echoSE:
        squareErrors = []

    policy = Policy(mdp)

    for i in range(len(policy.parameters)):
        policy.parameters[i] = InitParameter

    for _ in range(iterNum):
        if echoSE:
            squareErrors.append(getSquareErrorPolicy(policy))

        state = random.choice(mdp.states)
        sFeature = mdp.getFeature(state)
        action = random.choice(mdp.actions)
        isTerminal = False

        count = 0
        while not isTerminal and count < maxWalkLen:
            isTerminal, nextState, reward, nextSFeature = mdp.transform(state, action)

            maxQ = -1.0
            for nextAction in mdp.actions:
                q = policy.qFunc(nextSFeature, nextAction)
                if maxQ < q:
                    maxQ = q

            updatePolicy(policy, sFeature, action, reward + mdp.gamma * maxQ, alpha)

            action = policy.epsilonGreedy(nextSFeature, epsilon)
            state = nextState
            sFeature = nextSFeature
            count += 1

    if echoSE:
        return policy, squareErrors
    else:
        return policy


def test():
    getBestQFunc()

    plt.figure(figsize=(14, 10))

    iterNum = 10000
    epsilon = 0.5
    alpha = 0.01
    x = [i for i in range(iterNum)]

    mdpWall = FeatureMDP(feature='wall')
    mdpId = FeatureMDP(feature='identity')

    policy, se = featureMCControl(mdpWall, epsilon, alpha, iterNum, echoSE=True)
    plt.plot(x, se, '-', label='MC wall')

    policy, se = featureMCControl(mdpId, epsilon, alpha, iterNum, echoSE=True)
    plt.plot(x, se, '-', label='MC id')

    policy, se = featureSARSA(mdpWall, epsilon, alpha, iterNum, echoSE=True)
    plt.plot(x, se, '-', label='SARSA wall')

    policy, se = featureSARSA(mdpId, epsilon, alpha, iterNum, echoSE=True)
    plt.plot(x, se, '-', label='SARSA id')

    policy, se = featureQLearning(mdpWall, epsilon, alpha, iterNum, echoSE=True)
    plt.plot(x, se, '-', label='Q-Learning wall')

    policy, se = featureQLearning(mdpId, epsilon, alpha, iterNum, echoSE=True)
    plt.plot(x, se, '-', label='Q-Learning id')

    plt.xlabel(r'number of iterations')
    plt.ylabel(r'square errors')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test()
