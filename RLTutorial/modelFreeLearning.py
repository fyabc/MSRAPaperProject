#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
import random
import matplotlib.pyplot as plt

from MDP import MDP
import version23


__author__ = 'fyabc'

r"""
When the policy is model-free, we do not know Reward(s, a) and Transform(s, a),
so we use q(s, a).

Greedy policy:
    policy(s, a) =
        1           , if a = argmax_a(q(s, a))
        0           , else
Greedy policy may not be best, so we use \epsilon-greedy policy.

    policy_{n_greedy}(s, a) =
        1 - \epsilon + \epsilon / |A|       , a = argmax_a(q(s, a))
        \epsilon / |A|                      , else
"""


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
    # result = 0.0
    #
    # for key in qFunc:
    #     result += (qFunc[key] - bestQFunc[key]) ** 2
    #
    # return result


def epsilonGreedy(mdp, qFunc, state, epsilon):
    # argmax_a(q(s, a))
    maxAIndex = 0
    currentKey = state, mdp.actions[0]
    maxQ = qFunc[currentKey]

    for i in range(mdp.actionSize):
        currentKey = state, mdp.actions[i]
        q = qFunc[currentKey]

        if maxQ < q:
            maxQ = q
            maxAIndex = i

    temp = epsilon / mdp.actionSize

    # random choose
    choice = random.random()

    # get action chosen
    sumProb = 0.0
    for i in range(mdp.actionSize):
        if i == maxAIndex:
            sumProb += 1 - epsilon + temp
        else:
            sumProb += temp
        if sumProb >= choice:
            return mdp.actions[i]
    return mdp.actions[-1]


def greedy(mdp, qFunc, state):
    # argmax_a(q(s, a))
    maxAIndex = 0
    currentKey = state, mdp.actions[0]
    maxQ = qFunc[currentKey]

    for i in range(mdp.actionSize):
        currentKey = state, mdp.actions[i]
        q = qFunc[currentKey]

        if maxQ < q:
            maxQ = q
            maxAIndex = i

    return mdp.actions[maxAIndex]


def MCControl(mdp, epsilon, iterNum, maxWalkLen=100, echoSE=False):
    r"""
    for all (s, a) pair:
    obtain action value function q(s, a) and number visited n(s, a).

    MC Control algorithm generate a random walk, then update q and n from it.

    q(s, a) = (q(s, a) * n(s, a) + g) / (n(s, a) + 1)
    n(s, a) = n(s, a) + 1

    g = r_t + \gamma * r_{t+1} + ...    (the expected reward)
    """

    n = {
        (state, action): 0.0001  # smooth
        for state in mdp.states
        for action in mdp.actions
    }
    qFunc = {
        (state, action): 0.0
        for state in mdp.states
        for action in mdp.actions
    }

    if echoSE:
        squareErrors = []

    for _ in range(iterNum):
        if echoSE:
            squareErrors.append(getSquareError(qFunc))

        states, actions, rewards = [], [], []

        state = random.choice(mdp.states)
        isTerminal = False

        count = 0
        while not isTerminal and count < maxWalkLen:
            action = epsilonGreedy(mdp, qFunc, state, epsilon)
            isTerminal, nextState, reward = mdp.transform(state, action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = nextState
            count += 1

        g = 0.0
        for i in range(len(states) - 1, -1, -1):
            g *= mdp.gamma
            g += rewards[i]

        for i in range(len(states)):
            key = states[i], actions[i]
            qFunc[key] = (qFunc[key] * n[key] + g) / (n[key] + 1)
            n[key] += 1

            g -= rewards[i]
            g /= mdp.gamma

    if echoSE:
        return qFunc, squareErrors
    else:
        return qFunc


def SARSA(mdp, alpha, epsilon, iterNum, maxWalkLen=100, echoSE=False):
    r"""
    alpha: learning rate

    State Action Reward State Action (SARSA) is the action-reward version of TD algorithm.

    q(s, a) = q(s, a) + \alpha * (r + \gamma * q(s', a') - q(s, a))
    """

    qFunc = {
        (state, action): 0.0
        for state in mdp.states
        for action in mdp.actions
    }

    if echoSE:
        squareErrors = []

    for _ in range(iterNum):
        if echoSE:
            squareErrors.append(getSquareError(qFunc))

        state = random.choice(mdp.states)
        action = random.choice(mdp.actions)
        isTerminal = False

        count = 0
        while not isTerminal and count < maxWalkLen:
            key = state, action
            isTerminal, nextState, reward = mdp.transform(state, action)

            nextAction = epsilonGreedy(mdp, qFunc, nextState, epsilon)
            nextKey = nextState, nextAction

            qFunc[key] += alpha * (reward + mdp.gamma * qFunc[nextKey] - qFunc[key])

            state = nextState
            action = nextAction
            count += 1

    if echoSE:
        return qFunc, squareErrors
    else:
        return qFunc


def QLearning(mdp, alpha, epsilon, iterNum, maxWalkLen=100, echoSE=False):
    r"""
    alpha: learning rate

    q(s, a) = q(s, a) + \alpha * (r + \gamma * max_{a'}(q(s', a')) - q(s, a))
    """

    qFunc = {
        (state, action): 0.0
        for state in mdp.states
        for action in mdp.actions
    }

    if echoSE:
        squareErrors = []

    for _ in range(iterNum):
        if echoSE:
            squareErrors.append(getSquareError(qFunc))

        state = random.choice(mdp.states)
        action = random.choice(mdp.actions)
        isTerminal = False

        count = 0
        while not isTerminal and count < maxWalkLen:
            key = state, action
            isTerminal, nextState, reward = mdp.transform(state, action)

            nextKey = nextState, mdp.actions[0]
            maxQ = -1.0
            for nextAction in mdp.actions:
                if maxQ < qFunc[nextState, nextAction]:
                    maxQ = qFunc[nextState, nextAction]
                    nextKey = nextState, nextAction

            qFunc[key] += alpha * (reward + mdp.gamma * qFunc[nextKey] - qFunc[key])

            state = nextState
            action = epsilonGreedy(mdp, qFunc, nextState, epsilon)
            count += 1

    if echoSE:
        return qFunc, squareErrors
    else:
        return qFunc


def showQFunc(mdp, qFunc):
    print('=======')
    for state in mdp.states:
        for action in mdp.actions:
            print('%d %s: %2.1f' % (state, action, qFunc[state, action]))
    print()


###################
# Test Functions. #
###################


def testVariance():
    getBestQFunc()

    plt.figure(figsize=(14, 10))

    mdp = MDP()
    iterNum = 6000
    epsilon = 0.2
    alpha = 0.2
    x = [i for i in range(iterNum)]

    repeatNum = 1

    for i in range(repeatNum):
        qFunc, se = MCControl(mdp, epsilon, iterNum, echoSE=True)
        plt.plot(x, se, '-', label=r'mc $\epsilon = %2.1f$' % epsilon)
        # showQFunc(mdp, qFunc)

    for i in range(repeatNum):
        qFunc, se = SARSA(mdp, alpha, epsilon, iterNum, echoSE=True)
        plt.plot(x, se, '--', label=r'sarsa $\alpha = %2.1f, \epsilon = %2.1f$' % (alpha, epsilon))
        # showQFunc(mdp, qFunc)

    for i in range(repeatNum):
        qFunc, se = QLearning(mdp, alpha, epsilon, iterNum, echoSE=True)
        plt.plot(x, se, '-.', label=r'q-learning $\alpha = %2.1f, \epsilon = %2.1f$' % (alpha, epsilon))
        # showQFunc(mdp, qFunc)

    plt.xlabel(r'number of iterations')
    plt.ylabel(r'square errors')
    plt.legend()
    plt.show()


def testEpsilonGreedy():
    pass


def testLearningRate():
    pass


def testComprehensive():
    pass


def testNearOptimal():
    pass


def test():
    testVariance()


if __name__ == '__main__':
    test()
