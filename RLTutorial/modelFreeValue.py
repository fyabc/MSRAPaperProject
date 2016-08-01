#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
from MDP import MDP
import version23


__author__ = 'fyabc'


def MonteCarlo(mdp, stateSamples, actionSamples, rewardSamples):
    vFunc = {
        state: 0.0
        for state in mdp.states
    }
    nFunc = {
        state: 0
        for state in mdp.states
    }

    for i in range(len(stateSamples)):
        # G: total rewards
        G = 0.0
        expGamma = 1.0

        # Using every MC method
        for step in range(len(stateSamples[i])):
            G += rewardSamples[i][step] * expGamma
            expGamma *= mdp.gamma

            state = stateSamples[i][step]
            vFunc[state] += G
            nFunc[state] += 1

    for state in mdp.states:
        if nFunc[state] > 0:
            vFunc[state] /= nFunc[state]

    return vFunc


def temporalDifference(mdp, alpha, stateSamples, actionSamples, rewardSamples):
    # TD(0)
    # TD update: v(s) = v(s) + \alpha * (r + \gamma * v(s') - v(s))
    vFunc = {
        state: 0.0
        for state in mdp.states
    }

    for i in range(len(stateSamples)):
        for step in range(len(stateSamples[i])):
            state = stateSamples[i][step]
            reward = rewardSamples[i][step]

            if step < len(stateSamples[i]) - 1:
                nextState = stateSamples[i][step + 1]
                nextV = vFunc[nextState]
            else:
                nextV = 0.0

            vFunc[state] += alpha * (reward + mdp.gamma * nextV - vFunc[state])

    return vFunc


def test():
    mdp = MDP(0.5)

    vFunc = MonteCarlo(mdp, *mdp.randomWalk(100))

    print('Monte Carlo:')
    for i in range(1, 6):
        print('%d: %f\t' % (i, vFunc[i]), end='')
    print()

    vFunc = temporalDifference(mdp, 0.15, *mdp.randomWalk(100))

    print('Temporal Difference:')
    for i in range(1, 6):
        print('%d: %f\t' % (i, vFunc[i]), end='')
    print()


if __name__ == '__main__':
    test()
