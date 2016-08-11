#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import unicode_literals

__author__ = 'fyabc'

Config = {
    "environment": "CartPole-v0",
    "render": False,
    "retrain": False,
    "parameters": {
        "episode": 10000,       # Episode limitation
        "step": 300,            # Step limitation in an episode
        "test": 10,             # The number of experiment test every 100 episode

        # Hyper parameters for DQN
        "gamma": 0.9,           # discount factor for target Q
        "initEpsilon": 0.5,     # starting value of epsilon
        "finalEpsilon": 0.01,   # final value of epsilon
        "replaySize": 10000,    # experience replay buffer size
        "batchSize": 32         # size of mini-batch
    }
}

ParamConfig = Config["parameters"]
