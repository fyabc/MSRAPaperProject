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
        # init parameters
        self.timeStep = 0
        self.learningRate = ParamConfig['learningRate']
        self.epsilon = ParamConfig["initEpsilon"]
        self.gamma = ParamConfig['gamma']
        self.replaySize = ParamConfig['replaySize']
        self.batchSize = ParamConfig['batchSize']
        self.H = ParamConfig['H']

        self.stateSize = env.observation_space.shape[0]
        self.actionSize = env.action_space.n

        self.replayBuffer = deque(maxlen=self.replaySize)

        self.params = {
            'W1': None,
            'b1': None,
            'W2': None,
            'b2': None
        }

        # stateInput: (batchSize, stateSize)
        #
        # W1: (stateSize, H)
        # b1: (H)
        # hiddenOutput: (batchSize, H)
        #
        # W2: (H, actionSize)
        # b2: (actionSize)
        # QValue: (batchSize, actionSize)
        #
        # actionInput: (batchSize, actionSize)
        # targetQ: (batchSize)
        # cost: (1)

        self.stateInput = None
        self.QValue = None
        self.QFunction = None

        # one-hot presentation.
        self.actionInput = None

        # targetQ is the target Q value.
        self.targetQ = None

        self.cost = None
        self.sgd = None

        self.createQNetwork()
        self.createTrainingMethod()

    def createQNetwork(self):
        """
        Create a Q network.
        This is a very simple full-connected network with only one hidden layer.

        parameters to train are: W1, b1, W2, b2
        Q = W2 * ReLU(W1 * x + b1) + b2
        """

        self.params['W1'] = shared(name='W1', value=initNorm(self.stateSize, self.H))
        self.params['b1'] = shared(name='b1', value=np.ones((self.H,), dtype=fX) * ParamConfig['initB'])
        self.params['W2'] = shared(name='W2', value=initNorm(self.H, self.actionSize))
        self.params['b2'] = shared(name='b2', value=np.ones((self.actionSize,), dtype=fX) * ParamConfig['initB'])

        self.stateInput = T.matrix('stateInput', dtype=fX)
        hiddenOutput = T.nnet.relu(T.dot(self.stateInput, self.params['W1']) + self.params['b1'])
        self.QValue = T.dot(hiddenOutput, self.params['W2']) + self.params['b2']

        self.QFunction = function(
            inputs=[self.stateInput],
            outputs=self.QValue
        )

    def createTrainingMethod(self):
        """
        Create a training method.
        """

        self.actionInput = T.matrix('actionInput', dtype=fX)
        self.targetQ = T.vector('targetQ', dtype=fX)
        QActions = T.sum(self.QValue * self.actionInput, axis=1)
        self.cost = T.mean(T.square(self.targetQ - QActions))

        gParams = {paramName: T.grad(self.cost, self.params[paramName]) for paramName in self.params}

        updates = [(self.params[paramName], self.params[paramName] - self.learningRate * gParams[paramName])
                   for paramName in self.params]

        self.sgd = function(
            inputs=[self.stateInput, self.actionInput, self.targetQ],
            outputs=self.cost,
            updates=updates,
        )

    def perceive(self, state, action, reward, nextState, done):
        """
        perceive: 感知
        """

        oneHot = self.__makeOneHotAction(action)

        self.replayBuffer.append((state, oneHot, reward, nextState, done))

        if len(self.replayBuffer) > self.batchSize:
            self.trainQNetwork()

    def trainQNetwork(self):
        self.timeStep += 1

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayBuffer, self.batchSize)
        stateBatch = [data[0] for data in minibatch]
        actionBatch = [data[1] for data in minibatch]
        rewardBatch = [data[2] for data in minibatch]
        nextStateBatch = [data[3] for data in minibatch]

        # Step 2: calculate y
        yBatch = []
        QValueBatch = self.QFunction(np.asmatrix(nextStateBatch, dtype=fX))

        for i in range(self.batchSize):
            done = minibatch[i][4]
            if done:
                yBatch.append(rewardBatch[i])
            else:
                yBatch.append(rewardBatch[i] + self.gamma * np.max(QValueBatch[i]))

        self.sgd(
            stateInput=np.asmatrix(stateBatch, dtype=fX),
            actionInput=np.asmatrix(actionBatch, dtype=fX),
            targetQ=np.asarray(yBatch, dtype=fX),
        )

    def epsilonGreedyAction(self, state):
        QValue = self.QFunction(np.asmatrix(state, dtype=fX))[0]

        if random.random() <= self.epsilon:
            self.epsilon -= (ParamConfig['initEpsilon'] - ParamConfig['finalEpsilon']) / 10000
            return random.randint(0, self.actionSize - 1)
        else:
            self.epsilon -= (ParamConfig['initEpsilon'] - ParamConfig['finalEpsilon']) / 10000
            return np.argmax(QValue)

    def greedyAction(self, state):
        return np.argmax(self.QFunction(np.asmatrix(state, dtype=fX))[0])

    def __makeOneHotAction(self, action):
        oneHot = np.zeros(self.actionSize)
        oneHot[action] = 1

        return oneHot


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
            action = agent.epsilonGreedyAction(state)
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
            if avgReward >= ParamConfig['acceptThreshold']:
                print("The average reward has been high enough, iteration break!")
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


def miniTest():
    env = gym.make(Config["environment"])
    agent = DQN(env)
    for param in agent.params:
        print(param, agent.params[param].get_value())
    try:
        print('State size:', agent.stateSize)
        state = env.step(0)[0]
        nState = np.asarray(state, dtype=fX)
        print(agent.QFunction(np.matrix(state, dtype=fX)))
        print(agent.epsilonGreedyAction(nState))
    except Exception as e:
        print('==============================')
        print(e)


if __name__ == '__main__':
    main()
    # miniTest()
