#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import unicode_literals

__author__ = 'fyabc'

Config = {
    "environment": "CartPole-v0",
    "render": True,
    "retrain": False,
    "parameters": {
        "episode": 10000,       # Episode limitation
        "step": 300,            # Step limitation in an episode
        "test": 10,             # The number of experiment test every 100 episode

        # Hyper parameters for DQN
        "gamma": 0.9,           # discount factor for target Q
        "learningRate": 0.01,   # learning rate
        "initEpsilon": 0.5,     # starting value of epsilon
        "finalEpsilon": 0.01,   # final value of epsilon
        "replaySize": 10000,    # experience replay buffer size
        "batchSize": 32,        # size of mini-batch
        "H": 20,                # size (number of neurons) of hidden layer
        "initB": 0.1,           # init value of bias nodes
        "acceptThreshold": 295  # the threshold that we will accept the result
    }
}

ParamConfig = Config["parameters"]
