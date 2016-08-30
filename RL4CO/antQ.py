#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import numpy as np

__author__ = 'fyabc'


class Agent(object):
    def __init__(self, start_city, n):
        self.n = n
        self.start_city = start_city
        self.remain_cities = set(range(self.n)) - {self.start_city}
        self.location = start_city


def antQ(distances, agents):
    n = len(distances)

    # Initialize
    AQ0 = 0.
    AQ = np.ones((n, n), dtype='float64') * AQ0

    m = len(agents)

    # This is the step in which agents build their tours. The tour of agent k is stored in
    # Tourk. Given that local reinforcement is always null, only the next state evaluation is used
    # to update AQ-values.


def main():
    pass


if __name__ == '__main__':
    main()
