#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
import os
import json

__author__ = 'fyabc'

Config = json.load(open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r'))

if __name__ == '__main__':
    print(Config)
