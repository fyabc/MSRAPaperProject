#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
from MDP import MDP
import version23


__author__ = 'fyabc'


class PolicyLearner:
    def __init__(self, mdp):
        # The value of state i
        self.values = [0.0 for _ in range(mdp.STATE_NUM + 1)]

        # The action of state i
        self.policy = {
            state: mdp.actions[0]
            for state in mdp.states
            if state not in mdp.terminalStates
        }

        self.mdp = mdp

        self.maxIterNum = 1000
        self.epsilon = 1e-6

    def reset(self):
        self.values = [0.0 for _ in range(self.mdp.STATE_NUM + 1)]
        self.policy = {
            state: self.mdp.actions[0]
            for state in self.mdp.states
            if state not in self.mdp.terminalStates
        }

    def show(self):
        print('Value:')
        for i in range(1, 6):
            print('%d: %f\t' % (i, self.values[i]), end='')
        print()

        print('Policy:')
        for i in range(1, 6):
            print('%d->%s\t' % (i, self.policy[i]), end='')
        print()

        print()

    def policyEvaluate(self):
        for _ in range(self.maxIterNum):
            delta = 0.0

            for state in self.mdp.states:
                if state in self.mdp.terminalStates:
                    continue

                action = self.policy[state]
                isTerminal, nextState, reward = self.mdp.transform(state, action)

                newValue = reward + self.mdp.gamma * self.values[nextState]
                delta += abs(self.values[state] - newValue)
                self.values[state] = newValue

            if delta < self.epsilon:
                break

    def policyImprove(self):
        policyStable = True

        for state in self.mdp.states:
            if state in self.mdp.terminalStates:
                continue

            nextAction = self.mdp.actions[0]
            newValue = -1000000

            for action in self.mdp.actions:
                isTerminal, nextState, reward = self.mdp.transform(state, action)
                if newValue < reward + self.mdp.gamma * self.values[nextState]:
                    nextAction = action
                    newValue = reward + self.mdp.gamma * self.values[nextState]

            if nextAction != self.policy[state]:
                policyStable = False
            self.policy[state] = nextAction

        return policyStable

    def policyIterate(self, iterNum=100):
        for _ in range(iterNum):
            self.policyEvaluate()
            policyStable = self.policyImprove()

            if policyStable:
                break

    def valueIterate(self, iterNum=1000):
        for _ in range(iterNum):
            delta = 0.0

            for state in self.mdp.states:
                if state in self.mdp.terminalStates:
                    continue

                nextAction = self.mdp.actions[0]
                newValue = -100000

                for action in self.mdp.actions:
                    isTerminal, nextState, reward = self.mdp.transform(state, action)
                    if newValue < reward + self.mdp.gamma * self.values[nextState]:
                        nextAction = action
                        newValue = reward + self.mdp.gamma * self.values[nextState]

                delta += abs(self.values[state] - newValue)

                self.policy[state] = nextAction
                self.values[state] = newValue

            if delta < self.epsilon:
                break


def test():
    mdp = MDP(0.8)
    pi = PolicyLearner(mdp)

    pi.policyIterate(99)

    print('Policy Iteration:')
    pi.show()

    pi.reset()
    pi.valueIterate()

    print('Value Iteration:')
    pi.show()


if __name__ == '__main__':
    test()
