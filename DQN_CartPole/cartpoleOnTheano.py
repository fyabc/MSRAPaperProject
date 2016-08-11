#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals
import gym
import numpy as np
import random
from collections import deque

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

        self.createQNetwork()
        self.createTrainingMethod()

    def createQNetwork(self):
        pass

    def createTrainingMethod(self):
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
            action = None

        # Test every $ParamConfig["test"] episodes

    # Save results for uploading


if __name__ == '__main__':
    main()
