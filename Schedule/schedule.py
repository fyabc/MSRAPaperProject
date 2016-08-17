#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
import heapq

__author__ = 'fyabc'


class Scheduler(object):
    def __init__(self, performances, sizes, dependencies):
        """

        :param performances: performances of each machine.
        :param sizes: sizes of each job.
            time = size / performance
        :param dependencies: a DAG of dependencies.
            (a list of set, if j in dependencies[i], it means i -> j)
            if there is an edge i -> j in DAG, job j must start after i was finished.
        """

        self.performances = performances
        self.sizes = sizes
        self.dependencies = dependencies
        self.preprocess = {}

    @property
    def M(self):
        return len(self.performances)

    @property
    def N(self):
        return len(self.sizes)

    def getTLevels(self):
        tLevels = [0.0 for _ in range(self.N)]

        outDegrees = [len(depI) for depI in self.dependencies]
        outZero = {i for i in range(self.N) if outDegrees[i] == 0}

        reverseDep = [set() for _ in range(self.N)]
        for i, depI in enumerate(self.dependencies):
            for j in depI:
                reverseDep[j].add(i)

        while outZero:
            j = outZero.pop()
            tLevels[j] += self.sizes[j]

            for i in reverseDep[j]:
                if tLevels[j] > tLevels[i]:
                    tLevels[i] = tLevels[j]
                outDegrees[i] -= 1
                if outDegrees[i] == 0:
                    outZero.add(i)

        return tLevels

    def tLevel(self):
        if 'tLevels' not in self.preprocess:
            self.preprocess['tLevels'] = self.getTLevels()
        tLevels = [(tLevel, i) for i, tLevel in enumerate(self.preprocess['tLevels'])]
        tLevels.sort(reverse=True)

        finishTimes = [0.0 for _ in range(self.M)]
        result = []

        for tLevel in tLevels:
            job = tLevel[1]
            size = self.sizes[job]

            machine = -1
            minTime = 1000000000

            for m in range(self.M):
                finishTime = finishTimes[m] + size / self.performances[m]
                if finishTime < minTime:
                    machine = m
                    minTime = finishTime

            result.append((job, machine))
            finishTimes[machine] += size / self.performances[machine]

        finalTime = max(finishTimes)
        return finalTime, result


def test():
    p, k1, k2 = 20., 10, 15

    performances = [1. for _ in range(k2 - 1)] + [p]
    sizes = [1.] * (k1 + k2)
    dependencies = [{i + 1} for i in range(k1 - 1)] + [set()] + [{19} for _ in range(k2 - 1)] + [set()]

    scheduler = Scheduler(performances, sizes, dependencies)

    finalTime, schedule = scheduler.tLevel()
    print(finalTime, schedule)


if __name__ == '__main__':
    test()
