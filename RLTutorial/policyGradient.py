#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import numpy as np
import random
import matplotlib.pyplot as plt

from MDP import FeatureMDP
from policy import Policy
from utils import getSquareErrorPolicy, getBestQFunc
import version23


__author__ = 'fyabc'

r"""
Policy Gradient Method

The value function approximation use a model to fit the value function.
Policy gradient method use a model to fit the policy directly.

1. Policy to parameters
    1) Discrete Actions
        s -> \hat{s}
        f(\hat{s}, a)   (length = |A| * |\hat{s}|)
        w: parameters

        Softmax policy is:
        policy_w(\hat{s}, a) = exp(f(\hat{s}, a)' * w) / \sigma_{a' in A}(exp(f(\hat{s}, a')' * w))
        (the probability of taking action a when meeting \hat{s})

        Gradient:
        grad_w(log(policy_w(\hat{s}, a))) = f(\hat{s}, a) - \sigma_{a' in A}( policy_w(\hat{s}, a') * f(\hat{s}, a') )

    2) Continuous Actions
        Gauss policy is:
        policy_w(\hat{s}, a) = (1 / sqrt(2 * pi)) * exp(-1/2 * sqr(a - \hat{s}' * w))

        a: a real value

        Gradient:
        grad_w(log(policy_w(\hat{s}, a))) = (a - \hat{s}' * w) * \hat{s}

    There are 3 objective function to learn w:
        J_1(w) = V^{policy_w}(s1) = E_{policy_w}[v1]
        J_{avV}(w) = \sigma_s(d^{policy_w}(s) * V^{policy_w}(s))
        J_{avR}(w) = \sigma_s( d^{policy_w}(s) * \sigma_a(policy_w(s, a) * R(s, a)) )

        d^{policy_w} is the stable probability of policy_w.

    Theorem:
        For any differentiable policy_w(s, a),
        For any J in J_1, J_avR, J_avV / (1 - gamma):
            grad_w(J(w)) = E_{policy_w}[grad_w(log(policy_w(\hat{s}, a))) * q^{policy_w}(\hat{s}, a)]

2. Policy Gradient Algorithm
    According to the theorem above, we should compute grad_w(log(policy_w(\hat{s}, a))) and q(\hat{s}, a).
    Grad of policy can be computed. Then compute q function.

    1) MC Policy Gradient
        explore the environment and generate a list:
            s_1, a_1, r_1, ..., s_T, a_T, r_T
        let g_t = r_t + \gamma * r_{t+1} + ... to be q(s_t, a).

    2) Actor-Critic
        Actor update policy, critic update value.
        Critic can use SARSA or Q-learning.
"""


class SoftmaxPolicy(Policy):
    def __init__(self, featureMDP):
        super(SoftmaxPolicy, self).__init__(featureMDP)

    def pFunc(self, sFeature):
        """
        the policy function (distribution) policy_w(\hat{s})
        :return a list of probabilities for all actions in A.
        """
        prob = [0.0 for _ in range(self.mdp.actionSize)]
        normalizeFactor = 0.0

        for i in range(len(prob)):
            feature = self.getActionFeatureVector(sFeature, self.mdp.actions[i])
            prob[i] = np.exp(np.dot(feature, self.parameters))
            normalizeFactor += prob[i]

        for i in range(len(prob)):
            prob[i] /= normalizeFactor

        return prob

    def update(self, sFeature, action, bestQValue, alpha):
        """
        grad_w(log(policy_w(\hat{s}, a))) = f(\hat{s}, a) - \sigma_{a' in A}( policy_w(\hat{s}, a') * f(\hat{s}, a') )
        """
        feature = self.getActionFeatureVector(sFeature, action)
        prob = self.pFunc(sFeature)

        delta_logJ = np.asarray(feature, dtype='float64')

        for i in range(self.mdp.actionSize):
            delta_logJ -= self.getActionFeatureVector(sFeature, self.mdp.actions[i]) * prob[i]
        # delta_logJ = -delta_logJ

        self.parameters -= alpha * delta_logJ * bestQValue


def policyMCControl(mdp, epsilon, alpha, iterNum, maxWalkLen=100, echoSE=False):
    """
    qFunc = g_t
    """

    InitParameter = 0.1

    if echoSE:
        squareErrors = []

    policy = SoftmaxPolicy(mdp)

    for i in range(len(policy.parameters)):
        policy.parameters[i] = InitParameter

    for _ in range(iterNum):
        if echoSE:
            squareErrors.append(getSquareErrorPolicy(policy))

    if echoSE:
        return policy, squareErrors
    else:
        return policy


def policySARSA(mdp, epsilon, alpha, iterNum, maxWalkLen=100, echoSE=False):
    """
    Actor-Critic: actor update the policy, critic update the value.
    """

    InitParameter = 0.1

    if echoSE:
        squareErrors = []

    policy = SoftmaxPolicy(mdp)
    valuePolicy = Policy(mdp)

    for i in range(len(policy.parameters)):
        policy.parameters[i] = InitParameter
        valuePolicy.parameters[i] = 0.0

    for _ in range(iterNum):
        if echoSE:
            squareErrors.append(getSquareErrorPolicy(valuePolicy))

        state = random.choice(mdp.states)
        sFeature = mdp.getFeature(state)
        action = random.choice(mdp.actions)
        isTerminal = False

        count = 0
        while not isTerminal and count < maxWalkLen:
            isTerminal, nextState, reward, nextSFeature = mdp.transform(state, action)
            nextAction = policy.epsilonGreedy(nextSFeature, epsilon)

            valuePolicy.update(sFeature, action,
                               reward + mdp.gamma * valuePolicy.qFunc(nextSFeature, nextAction), alpha)
            policy.update(sFeature, action, valuePolicy.qFunc(sFeature, action), alpha)

            sFeature = nextSFeature
            action = nextAction
            count += 1

    if echoSE:
        return policy, squareErrors
    else:
        return policy


def policyQLearning(mdp, epsilon, alpha, iterNum, maxWalkLen=100, echoSE=False):
    pass


def test():
    getBestQFunc()

    iterNum = 6000
    epsilon = 0.2
    alpha = 0.01
    x = [i for i in range(iterNum)]

    mdpId = FeatureMDP(feature='identity')

    policy, se = policySARSA(mdpId, epsilon, alpha, iterNum, echoSE=True)
    plt.plot(x, se, '-', label='SARSA')

    plt.xlabel(r'number of iterations')
    plt.ylabel(r'square errors')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test()
