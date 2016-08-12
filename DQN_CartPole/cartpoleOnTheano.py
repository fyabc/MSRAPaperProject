#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
import gym
import numpy as np
import random
from collections import deque

import theano
from theano import shared, function
import theano.tensor as T

from config import Config, ParamConfig

__author__ = 'fyabc'

r"""
The (basic) DQN algorithm [NIPS 13](http://arxiv.org/pdf/1312.5602v1.pdf):

Algorithm 1: Deep Q-learning with Experience Replay:
    Initialize replay memory D to capacity N
    Initialize action-value function Q with random weights

    for episode = 1 to M do:
        Initialize sequence s_1 = {x_1} and (optional) preprocessed \phi_1 = \phi(s_1)

        for t = 1 to T do:
            With probability \epsilon:  select a random action a_t
            Otherwise:                  select a_t = max_a(Q*(\phi(s_t), a; \theta)

            Execute action a_t in emulator and observe reward r_t and image x_{t+1}

            Set s_{t+1} = s_t, a_t, x_{t+1} and preprocess \phi_{t+1} = \phi(s_{t+1})

            Store transition (\phi_t, a_t, r_t, \phi_{t+1}) in D

            # Because samples are relatively in time, so we do not use (t+1) to update, use a random (j+1) instead.
            Sample random mini-batch of transitions (\phi_j, a_j, r_j, \phi_{j+1}) from D

            Set y_j =   r_j                                                 for terminal \phi_{j+1}
                        r_j + \gamma * max_{a'}(Q(\phi_{j+1}, a'; \theta))  for non-terminal \phi_{j+1}

            # according to the equation:
            #   L(w) = E[(r + \gamma * max_{a'}(Q(s', a', w)) - Q(s, a, w)) ^ 2]
            #             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-------------------------> This part is target

            Perform a gradient descent step on (y_j - Q(\phi_j, a_j; \theta))^2
"""

fX = theano.config.floatX


def initNorm(nIn, nOut):
    return np.asarray(np.random.randn(nIn, nOut) * 0.1, dtype=fX)


class DQN(object):
    """
    DQN Agent.
    """
    def __init__(self, env):
        self.replayBuffer = deque()

        # init parameters
        self.timeStep = 0
        self.epsilon = ParamConfig["initEpsilon"]
        self.stateSize = env.observation_space.shape[0]
        self.actionSize = env.action_space.n
        self.params = {
            'W1': None,
            'b1': None,
            'W2': None,
            'b2': None
        }
        self.stateInput = None
        self.actionInput = None
        self.labelInput = None
        self.QValue = None
        self.QFunction = None

        self.createQNetwork()
        self.createTrainingMethod()

    def createQNetwork(self):
        """
        Create a Q network.
        This is a very simple full-connected network with only one hidden layer.

        parameters to train are: W1, b1, W2, b2
        Q = W2 * ReLU(W1 * x + b1) + b2
        """

        self.params['W1'] = shared(name='W1', value=initNorm(ParamConfig['H'], self.stateSize))
        self.params['b1'] = shared(name='b1', value=np.ones((ParamConfig['H'],), dtype=fX) * ParamConfig['initB'])
        self.params['W2'] = shared(name='W2', value=initNorm(self.actionSize, ParamConfig['H']))
        self.params['b2'] = shared(name='b2', value=np.ones((self.actionSize,), dtype=fX) * ParamConfig['initB'])

        self.stateInput = T.vector('x', dtype=fX)
        hiddenOutput = T.nnet.relu(T.dot(self.params['W1'], self.stateInput) + self.params['b1'])
        self.QValue = T.dot(self.params['W2'], hiddenOutput) + self.params['b2']

        self.QFunction = function(
            inputs=[self.stateInput],
            outputs=self.QValue,
        )

    def createTrainingMethod(self):
        """
        Create a training method.
        """

        self.actionInput = None
        self.labelInput = None

    def perceive(self, state, action, reward, nextState, done):
        """
        perceive: 感知
        """

        pass

    def trainQNetwork(self):
        pass

    def epsilonGreedyAction(self, state):
        pass

    def greedyAction(self, state):
        pass


def main():
    # Initialize OpenAI Gym env and DQN agent
    env = gym.make(Config["environment"])
    agent = DQN(env)

    for episode in range(ParamConfig["episode"]):
        # Initialize task
        state = env.reset()

        # Train
        for step in range(ParamConfig["step"]):
            # epsilon-greedy action for train
            action = agent.greedyAction(state)
            nextState, reward, done, _ = env.step(action)

            # define reward for agent
            rewardAgent = -1 if done else 0.1
            agent.perceive(state, action, reward, nextState, done)
            state = nextState
            if done:
                break

        # Test every $ParamConfig["test"] episodes
        # test is like train, using greedy action instead.
        if episode % ParamConfig["test"] == 0:
            totalReward = 0
            for i in xrange(ParamConfig['test']):
                state = env.reset()
                for j in xrange(ParamConfig['step']):
                    if Config['render']:
                        env.render()

                    # direct action for test
                    action = agent.greedyAction(state)
                    state, reward, done, _ = env.step(action)
                    totalReward += reward
                    if done:
                        break
            avgReward = totalReward / ParamConfig['test']

            print('Episode:', episode, 'Evaluation Average Reward:', avgReward)
            if avgReward >= 200:
                break

    # # Save results for uploading
    # env.monitor.start('gym_results/%s-experiment-1' % Config['environment'], force=True)
    # for i in xrange(100):
    #     state = env.reset()
    #     for j in xrange(200):
    #         if Config['render']:
    #             env.render()
    #
    #         # direct action for test
    #         action = agent.greedyAction(state)
    #         state, _, done, _ = env.step(action)
    #         if done:
    #             break
    # env.monitor.close()


if __name__ == '__main__':
    # main()
    env = gym.make(Config["environment"])
    agent = DQN(env)
    for param in agent.params:
        print(param, agent.params[param].get_value())
    try:
        print('State size:', agent.stateSize)
        print(agent.QFunction(np.asarray(env.step(0)[0], dtype=fX)))
    except Exception as e:
        print('==============================')
        print(e)
