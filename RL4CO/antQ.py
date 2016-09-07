#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import numpy as np

__author__ = 'fyabc'


Config = {
    'dataset': 'att48',

    'delayed_reinforcement': 'iteration_best',
    'floatX': 'float64',

    'iteration_num': 300,

    'parameters': {
        'delta': 1.,
        'beta': 2.,
        'q_0': 0.9,
        'alpha': 0.1,
        'gamma': 0.3,
        'W': 10,
    }
}

Config['distances_file'] = 'data/{}_d.txt'.format(Config['dataset'])
Config['best_result_file'] = 'data/{}_s.txt'.format(Config['dataset'])

fX = Config['floatX']

ParamConfig = Config['parameters']


def get_best_result(distances, filename=Config['best_result_file']):
    result = 0.

    cities = map(lambda l: int(l.strip()) - 1, list(open(filename, 'r')))
    for i in range(len(cities) - 1):
        result += distances[cities[i], cities[i + 1]]

    return result, cities


def unlimited_range(start=0, step=1):
    while True:
        yield start
        start += step


class Agent(object):
    def __init__(self, start_city, n):
        self.n = n

        # Let r_k1 be the starting city for agent k
        self.start_city = start_city

        # J_k(r_k1) is the set of yet to be visited cities for agent k in city r_k1
        self.remain_cities = set(range(self.n)) - {self.start_city}

        # r_k is the city where agent k is located
        self.location = start_city

        # s_k is the next city
        self.next_city = 0

        # Tour_k
        self.tour = []

    def reset(self):
        self.remain_cities = set(range(self.n)) - {self.start_city}
        self.location = self.start_city
        self.next_city = 0
        self.tour = []

    def step_city(self):
        self.tour.append((self.location, self.next_city))
        self.location = self.next_city

    def get_length(self, distances):
        return sum(map(lambda e: distances[e], self.tour))

    def repr_tour(self):
        return [pair[0] for pair in self.tour] + [self.start_city]


def antQ(distances, agents):
    # 0.
    # Load hyperparameters
    n = distances.shape[0]

    delta = ParamConfig['delta']
    beta = ParamConfig['beta']
    q_0 = ParamConfig['q_0']
    alpha = ParamConfig['alpha']
    gamma = ParamConfig['gamma']
    W = ParamConfig['W']

    # 1.
    # Initialize
    AQ0 = 1. / (n * np.sum(distances) / ((n - 1) ** 2))
    AQ = np.ones((n, n), dtype=fX) * AQ0

    HE = 1. / distances

    total_best_length_idx = 0
    total_best_length = np.sum(distances)
    total_best_tour = None

    best_lengths_record = []

    for iteration in unlimited_range(1):
        # print('[Iteration {}]'.format(iteration))

        for agent in agents:
            agent.reset()

        # 2.
        # This is the step in which agents build their tours.
        # The tour of agent k is stored in Tour_k.
        # Given that local reinforcement is always null,
        # only the next state evaluation is used to update AQ-values.
        for i in range(n):
            if i != n - 1:
                for agent in agents:
                    r = agent.location
                    values = [
                        (AQ[r, u] ** delta) * (HE[r, u] ** beta) if u in agent.remain_cities else 0.
                        for u in range(n)
                    ]

                    # Choose the next city s_k according to formula(1)
                    if np.random.random() < q_0:
                        agent.next_city = np.argmax(values)
                    else:
                        agent.next_city = np.argmax(np.random.multinomial(1, values / np.sum(values)))

                    agent.remain_cities.discard(agent.next_city)
                    if i == n - 2:
                        agent.remain_cities.add(agent.start_city)
            else:
                # In this cycle all the agents go back to the initial city r_k1
                for agent in agents:
                    agent.next_city = agent.start_city

            for agent in agents:
                # Calculate AQ
                r_k, s_k = agent.location, agent.next_city

                # Formula(2)
                # where the reinforcement \delta_AQ(r_k,s_k) is always null
                AQ[r_k, s_k] = (1 - alpha) * AQ[r_k, s_k] +\
                    alpha * gamma * max(map(lambda z: AQ[s_k, z], agent.remain_cities))

                # walk one step
                agent.step_city()

        # 3.
        # In this step delayed reinforcement is computed and AQ - values are updated using formula (2),
        # in which the next state evaluation term gamma * Max(AQ(r_k1, z)) is null for all z.

        # Calculate L_k (the length of the tour done by agent[k])
        lengths = np.asarray([agent.get_length(distances) for agent in agents], dtype=fX)

        deltaAQ = np.zeros_like(AQ)
        # Compute \delta_AQ

        if Config['delayed_reinforcement'] == 'global_best':
            # Global-best
            best_agent = total_best_length_idx
        else:
            # Iteration-best
            best_agent = np.argmin(lengths)
        for pair in agents[best_agent].tour:
            deltaAQ[pair] = W / lengths[best_agent]

        # Update AQ applying a formula(2)
        AQ = (1 - alpha) * AQ + alpha * deltaAQ

        # 4.
        # If End_condition == True
        # print shortest of L_k and break
        best_length_idx = np.argmin(lengths)
        best_length = lengths[best_length_idx]
        best_lengths_record.append(best_length)
        # print('Best length is:', best_length)
        # print('Total best length is:', total_best_length)
        # print('Total best tour is:', total_best_tour)

        if best_length < total_best_length:
            print('[Iteration {}] down to {}'.format(iteration, best_length))
            total_best_length = best_length
            total_best_tour = agents[best_length_idx].repr_tour()
            total_best_length_idx = best_length_idx

        if iteration == Config['iteration_num']:
            print('Total best tour is:', total_best_tour)
            break

    return best_lengths_record


def main():
    distances = np.genfromtxt(Config['distances_file'], delimiter=str(' '))
    n = distances.shape[0]

    best_result, cities = get_best_result(distances)

    print('Best result:', best_result, cities)

    agents = [Agent(c, n) for c in range(n)]

    best_lengths_record = antQ(distances, agents)

    import matplotlib.pyplot as plt
    plt.plot(best_lengths_record)
    plt.show()


if __name__ == '__main__':
    main()
