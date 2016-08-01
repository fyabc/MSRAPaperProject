#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
from MDP import MDP
import version23


__author__ = 'fyabc'

'''
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
'''


def MCControl(mdp, epsilon, iterNum):
    """
    for all (s, a) pair:
    obtain action value function q(s, a) and number visited n(s, a).

    MC Control algorithm generate a random walk, then update q and n from it.

    q(s, a) = (q(s, a) * n(s, a) + g) / (n(s, a) + 1)
    n(s, a) = n(s, a) + 1

    g = r_t + \gamma * r_{t+1} + ...    (the expected reward)
    """
    n = {

    }
    qFunc = {

    }


def test():
    pass


if __name__ == '__main__':
    test()
