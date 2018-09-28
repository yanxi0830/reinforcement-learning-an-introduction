#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Artem Oboturov(oboturov@gmail.com)                             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class Bandit:
    def __init__(self, k_arm=10, epsilon=0, initial=0, step_size=0.1, sample_averages=False, USB_param=None,
                 gradient=False, gradient_baseline=False, true_reward = 0.):
        """
        Initialize Bandit Problem to run different algorithms
        :param k_arm: number of arms
        :param epsilon: probability for exploration in epsilon-greedy algorithm
        :param initial: initial estimation for each action
        :param step_size: constant step size for updating estimations
        :param sample_averages: if True, use sample-averages to update estimations instead of constant step size
        :param USB_param: if not None, use UCB algorithm to select action
        :param gradient: if True, use gradient based bandit algorithm
        :param gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
        :param true_reward:
        """
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)    # array([0, 1, ..., k-1])
        self.time = 0
        self.UCB_param = USB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial

        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward

        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # number of chosen times for each action
        self.action_count = np.zeros(self.k)

        # action with max estimation action-value
        self.best_action = np.argmax(self.q_true)

    def reset(self):
        self.q_true = np.random.randn(self.k) + self.true_reward
        self.q_estimation = np.zeros(self.k) + self.initial
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)

    def act(self):
        """
        Get an action for this bandit
        :return: action selected using epsilon-greedy
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        # TODO: UCB algo

        # TODO: gradient algo

        return np.argmax(self.q_estimation)

    def step(self, action):
        """
        Take an action, update estimation for this action
        :param action:
        :return: reward
        """
        # generate the reward under normal distribution N(mean=real reward, variance=1)
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        # previous times * previous average + current reward / time
        self.average_reward = ((self.time - 1.0) / self.time)*self.average_reward + (reward / self.time)
        self.action_count[action] += 1

        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += 1.0 / self.action_count[action] * (reward - self.q_estimation[action])
        elif self.gradient:
            # TODO
        else:
            # update estimation with constant step size
            self.q_estimation += self.step_size * (reward - self.q_estimation[action])

        return reward


