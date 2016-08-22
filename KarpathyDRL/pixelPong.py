#! /usr/bin/python
# -*- encoding: utf-8 -*-

""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """

from __future__ import print_function, unicode_literals

import sys
import numpy as np
import cPickle as pickle
import gym

__author__ = 'fyabc'

Config = {
    'resume': 'retrain' not in sys.argv,  # resume from previous checkpoint?
    'render': 'render' in sys.argv,  # render the game?

    # hyper-parameters
    'parameters': {
        'D': 80 * 80,  # input dimensionality: 80x80 grid
        'H': 200,  # number of hidden layer neurons
        'batch_size': 10,  # number of hidden layer neurons
        'learning_rate': 1e-4,  # every how many episodes to do a param update?
        'gamma': 0.99,  # discount factor for reward
        'decay_rate': 0.99,  # decay factor for RMSProp leaky sum of grad^2
    }
}

ParamConfig = Config['parameters']

resume = Config['resume']
render = Config['render']

# hyper-parameters
D = ParamConfig['D']
H = ParamConfig['H']
batch_size = ParamConfig['batch_size']
learning_rate = ParamConfig['learning_rate']
gamma = ParamConfig['gamma']
decay_rate = ParamConfig['decay_rate']

# model initialization
if resume:
    try:
        model = pickle.load(open('save.p', 'rb'))
    except IOError:
        model = {
            'W1': np.random.randn(H, D) / np.sqrt(D),
            'W2': np.random.randn(H) / np.sqrt(H),
        }
else:
    model = {
        'W1': np.random.randn(H, D) / np.sqrt(D),
        'W2': np.random.randn(H) / np.sqrt(H),
    }


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def pre_process(I):
    """ pre-process 210x160x3 uInt8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # down_sample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(inputLayer):
    hiddenState = np.dot(model['W1'], inputLayer)
    hiddenState[hiddenState < 0] = 0  # ReLU non-linearity
    logP = np.dot(model['W2'], hiddenState)
    p = sigmoid(logP)
    return p, hiddenState  # return probability of taking action 2, and hidden state


def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # back-pro prelu
    dW1 = np.dot(dh.T, ep_x)
    return {'W1': dW1, 'W2': dW2}


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

total_grad = {k: np.zeros_like(v) for k, v in model.iteritems()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}  # rmsprop memory

highest_reward = -100

while True:
    if render:
        env.render()

    # pre-process the observation, set input to network to be difference image
    cur_x = pre_process(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

    # record various intermediates (needed later for back-propagation)
    xs.append(x)  # observation
    hs.append(h)  # hidden state
    y = 1 if action == 2 else 0  # a "fake label"

    # grad that encourages the action that was taken to be taken
    # (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    dlogps.append(y - aprob)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    # if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
    #     print('ep %d: game finished, reward: %.1f' % (episode_number, reward) +\
    #           ('' if reward == -1 else ' !!!!!!!!'))

    if done:  # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        ep_x = np.vstack(xs)
        ep_h = np.vstack(hs)
        ep_dlogp = np.vstack(dlogps)
        ep_r = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_ep_r = discount_rewards(ep_r)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_ep_r -= np.mean(discounted_ep_r)
        discounted_ep_r /= np.std(discounted_ep_r)

        ep_dlogp *= discounted_ep_r  # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(ep_h, ep_dlogp)
        for k in model:
            total_grad[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for key, value in model.iteritems():
                g = total_grad[key]  # gradient
                rmsprop_cache[key] = decay_rate * rmsprop_cache[key] + (1 - decay_rate) * g ** 2
                model[key] += learning_rate * g / (np.sqrt(rmsprop_cache[key]) + 1e-5)
                total_grad[key] = np.zeros_like(value)  # reset batch gradient buffer

        observation = env.reset()  # reset env
        prev_x = None

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        if running_reward > highest_reward:
            highest_reward = running_reward

        print('resetting env. episode %d reward total was %f. running mean: %f. highest: %f'
              % (episode_number, reward_sum, running_reward, highest_reward))
        if episode_number % 100 == 0:
            print('Saving current model at episode %d...' % episode_number, end='')
            pickle.dump(model, open('save.p', 'wb'))
            print('done')

        reward_sum = 0
