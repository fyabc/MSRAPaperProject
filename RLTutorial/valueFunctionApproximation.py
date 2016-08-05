#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import random

from MDP import MDP, FeatureMDP
from policy import Policy
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
            policy.update(sFeatures[i], actions[i], g, alpha)

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

            policy.update(sFeature, action, reward + mdp.gamma * policy.qFunc(nextSFeature, nextAction), alpha)

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

            policy.update(sFeature, action, reward + mdp.gamma * maxQ, alpha)

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
