#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
from operator import gt

import version23

__author__ = 'fyabc'


def argmax(seq, cmpFunc=gt):
    if len(seq) == 0:
        return None

    maxValue = seq[0]
    maxIndex = 0

    for i, elem in seq:
        if cmpFunc(elem, maxValue):
            maxValue = elem
            maxIndex = i

    return maxIndex
