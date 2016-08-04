#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import random
import numpy as np

from MDP import MDP
import version23


__author__ = 'fyabc'


class Policy(object):
    def __init__(self, featureMDP):
        self.mdp = featureMDP

        # parameters is w
        self.parameters = np.zeros((self.mdp.actionSize * self.mdp.sFeatureSize,), dtype='float64')

    def getActionFeatureVector(self, sFeature, action):
        result = np.zeros_like(self.parameters, dtype='int32')

        sfSize = self.mdp.sFeatureSize
        for index, a in enumerate(self.mdp.actions):
            if a == action:
                result[index * sfSize:(index + 1) * sfSize] = sFeature
                break

        return result

    def qFunc(self, sFeature, action):
        return np.dot(self.getActionFeatureVector(sFeature, action), self.parameters)

    def epsilonGreedy(self, sFeature, epsilon):
        # argmax_a(q(\hat{s}, a))
        maxAIndex = 0
        maxQ = self.qFunc(sFeature, self.mdp.actions[0])

        for i in range(self.mdp.actionSize):
            a = self.mdp.actions[i]
            q = self.qFunc(sFeature, a)
            if maxQ < q:
                maxQ = q
                maxAIndex = i

        temp = epsilon / self.mdp.actionSize

        # random choose
        choice = random.random()

        # get action chosen
        sumProb = 0.0
        for i in range(self.mdp.actionSize):
            if i == maxAIndex:
                sumProb += 1 - epsilon + temp
            else:
                sumProb += temp
            if sumProb >= choice:
                return self.mdp.actions[i]
        return self.mdp.actions[-1]

    def update(self, sFeature, action, bestQValue, alpha):
        """
        alpha: learning rate.
        """
        qValue = self.qFunc(sFeature, action)
        error = qValue - bestQValue

        feature = self.getActionFeatureVector(sFeature, action)
        self.parameters -= alpha * error * feature
