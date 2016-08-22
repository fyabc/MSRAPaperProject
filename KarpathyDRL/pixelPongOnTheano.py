#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import unicode_literals, print_function
import sys
import pickle as pkl
import numpy as np

import theano.tensor as T
from theano import function, shared, config
import gym

__author__ = 'fyabc'

"""
Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym.

Policy Network:
    h = W1 .* x
    logP = W2 .* ReLU(h)
    p = sigmoid(logP)
"""

fX = config.floatX

Config = {
    'environment': 'Pong-v0',
    'render': 'render' in sys.argv,
    'retrain': 'retrain' in sys.argv,
    'D': 80 * 80,   # input shape: (80*80,)
    'H': 200,       # number of hidden layer neurons
}
D, H = Config['D'], Config['H']

previous = None


def toFX(value):
    return eval('np.%s(value)' % fX)


def preprocess(ob):
    """preprocess the observation."""
    ob = ob[35:195]         # crop
    ob = ob[::2, ::2, 0]    # down_sample by factor of 2
    ob[ob == 144] = 0       # erase background (background type 1)
    ob[ob == 109] = 0       # erase background (background type 2)
    ob[ob != 0] = 1         # everything else (paddles, ball) just set to 1
    ob = ob.astype(fX).ravel()

    global previous
    result = ob - previous if previous is not None else np.zeros(D)
    previous = ob

    return result


class PolicyNetwork(object):
    def __init__(self, retrain=False):
        # x:  (D)
        # W1: (H, D)
        # h:  (H)
        # W2: (H)
        # logP, p: (1)

        if retrain:
            W1Value = np.random.randn(H, D) / np.sqrt(D)
            W2Value = np.random.randn(H) / np.sqrt(H)
        else:
            W1Value, W2Value = self.load()

        self.W1 = shared(name='W1', value=W1Value)
        self.W2 = shared(name='W2', value=W2Value)
        self.parameters = [self.W1, self.W2]

        self.x = T.vector('x', dtype=fX)

        self.h = T.dot(self.W1, self.x)
        self.h = T.nnet.relu(self.h)

        logP = T.dot(self.W2, self.h)
        self.p = T.nnet.sigmoid(logP)

        self.policyFunction = function(
            inputs=[self.x],
            outputs=self.p
        )

    def dump(self):
        model = [parameter.get_value() for parameter in self.parameters]
        pkl.dump(model, open('save.p', 'wb'))

    @staticmethod
    def load():
        return pkl.load(open('save.p', 'rb'))

    def takeAction(self, x):
        pAction = self.policyFunction(x)
        return pAction, (2 if np.random.uniform() < pAction else 3)


def test():
    network = PolicyNetwork(retrain=Config['retrain'])

    env = gym.make(Config['environment'])
    ob = env.reset()

    episodeNumber = 0
    xs, rewards = [], []
    totalReward = 0

    running = True
    while running:
        if Config['render']:
            env.render()

        # pre-process the observation, set input to network to be difference image
        x = preprocess(ob)

        # forward the policy network and sample an action from the returned probability
        pAction, action = network.takeAction(x)

        # record various intermediates (needed later for back propagate)
        xs.append(x)  # observation

        # step the environment and get new measurements
        observation, reward, done, _ = env.step(action)

        # record reward (has to be done after we call step() to get reward for previous action)
        totalReward += reward
        rewards.append(reward)

        # an episode finished
        if done:
            episodeNumber += 1

            # TODO


if __name__ == '__main__':
    test()
