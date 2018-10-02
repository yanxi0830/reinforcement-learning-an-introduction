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
    def __init__(self, k_arm=10, epsilon=0.0, initial=0, step_size=0.1, sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, true_reward=0.0, nonstationary=False):
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
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial
        self.nonstationary = nonstationary

        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward

        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # number of chosen times for each action
        self.action_count = np.zeros(self.k)

        # action with max estimation action-value
        self.best_action = np.argmax(self.q_true)

        # soft-max distribution of each action for gradient algo
        self.action_prob = np.zeros(self.k)

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
        # epsilon-greedy, choose a random action with probability epsilon
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        # UCB algo
        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
                             self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice([action for action, q in enumerate(UCB_estimation) if q == q_best])

        # gradient algo
        if self.gradient:
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)

        return np.argmax(self.q_estimation)

    def step(self, action):
        """
        Take an action, update estimation for this action
        :param action: action in [0, 1, ..., k-1]
        :return: actual reward got from taking action
        """
        if self.nonstationary:
            # All the real_reward start out equal and then take independent random
            # walks (by adding a normally distributed increment with mean zero,
            # and standard deviation 0.01 to all the q*(a) on each step).
            self.q_true += 0.01 * np.random.randn()

        # generate the reward under N(real_reward, 1)
        reward = np.random.randn() + self.q_true[action]

        self.time += 1
        self.average_reward = (self.time - 1.0) / self.time * self.average_reward + reward / self.time
        self.action_count[action] += 1

        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += 1.0 / self.action_count[action] * (reward - self.q_estimation[action])
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            self.q_estimation = self.q_estimation + self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])

        return reward


def simulate(runs, time, bandits):
    """
    Simulate runs
    :param runs: number of full runs
    :param time: number of time-steps per run
    :param bandits: bandits for a run, containing which algo to run
    :return: best_action_counts, rewards
    """
    best_action_counts = np.zeros((len(bandits), runs, time))
    rewards = np.zeros(best_action_counts.shape)
    for i, bandit in enumerate(bandits):
        for r in tqdm(range(runs)):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1

    # sum best action counts and rewards over a run
    best_action_counts = best_action_counts.mean(axis=1)
    rewards = rewards.mean(axis=1)
    return best_action_counts, rewards


def figure_2_1():
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Reward Distribution")
    plt.savefig('../images/figure_2_1.png')
    plt.show()


def figure_2_2(runs=2000, time=1000):
    """
    Average performance of epsilon-greedy action-value methods on the 10-armed testbed.
    All methods used sample averages as their action-value estimates.
    """
    epsilons = [0, 0.1, 0.01]
    bandits = [Bandit(epsilon=eps, sample_averages=True) for eps in epsilons]
    best_action_counts, rewards= simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label='epsilon = %.02f' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='epsilon = %.02f' % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('../images/figure_2_2.png')
    plt.show()


def figure_2_3(runs=2000, time=1000):
    """
    The effect of optimistic initial action-value estimates on the 10-armed testbed.
    Both methods used a constant step-size parameter, alpha=0.1
    """
    bandits = []
    bandits.append(Bandit(epsilon=0, initial=5, step_size=0.1))
    bandits.append(Bandit(epsilon=0.1, initial=0, step_size=0.1))
    best_action_counts, _ = simulate(runs, time, bandits)

    plt.plot(best_action_counts[0], label='epsilon = 0, q = 5')
    plt.plot(best_action_counts[1], label='epsilon = 0.1, q = 0')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()

    plt.savefig('../images/figure_2_3.png')
    plt.show()


def figure_2_4(runs=2000, time=1000):
    """
    Average performance of UCB action selection on 10-armed testbed. As shown,
    UCB generally performs better than epsilon-greedy action selection, except in the first k steps,
    when its selected randomly among the as-yet-untried actions
    """
    bandits = []
    bandits.append(Bandit(epsilon=0, UCB_param=2, sample_averages=True))
    bandits.append(Bandit(epsilon=0.1, sample_averages=True))
    _, average_rewards = simulate(runs, time, bandits)

    plt.plot(average_rewards[0], label='UCB c = 2')
    plt.plot(average_rewards[1], label='epsilon greedy epsilon = 0.1')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()

    plt.savefig('../images/figure_2_4.png')
    plt.show()


def figure_2_5(runs=2000, time=1000):
    """
    Average performance of the gradient bandit algorithm with and without a reward baseline on the 10-armed testbed
    when the q*(a) are chosen to be near +4 rather tha near zero
    """
    bandits = []
    bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=False, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.4, gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.4, gradient_baseline=False, true_reward=4))
    best_action_counts, _ = simulate(runs, time, bandits)
    labels = ['alpha = 0.1, with baseline',
              'alpha = 0.1, without baseline',
              'alpha = 0.4, with baseline',
              'alpha = 0.4, without baselin']

    for i in range(0, len(bandits)):
        plt.plot(best_action_counts[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.legend()

    plt.savefig('../images/figure_2_5.png')
    plt.close()


def figure_2_6(runs=2000, time=1000):
    """
    A parameter study of the various bandit algorithms presented in this chapter.
    Each point is the average reward obtained over 1000 steps with a particular algorithm
    at a particular setting of its parameter.
    """
    labels = ['epsilon-greedy', 'gradient bandit',
              'UCB', 'optimistic initialization']
    generators = [lambda epsilon: Bandit(epsilon=epsilon, sample_averages=True),
                  lambda alpha: Bandit(gradient=True, step_size=alpha, gradient_baseline=True),
                  lambda coef: Bandit(epsilon=0, UCB_param=coef, sample_averages=True),
                  lambda initial: Bandit(epsilon=0, initial=initial, step_size=0.1)]
    parameters = [np.arange(-7, -1, dtype=np.float),
                  np.arange(-5, 2, dtype=np.float),
                  np.arange(-4, 3, dtype=np.float),
                  np.arange(-2, 3, dtype=np.float)]

    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))

    _, average_rewards = simulate(runs, time, bandits)
    rewards = np.mean(average_rewards, axis=1)

    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i:i + l], label=label)
        i += l
    plt.xlabel('Parameter(2^x)')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('../images/figure_2_6.png')
    plt.close()


def exercise_2_5(runs=2000, time=10000):
    """
    Design and conduct an experiment to demonstrate the difficulties that sample-avaerage methods
    have for nonstationary problems. Use a modified version of the 10-armed testbed in which
    all the q*(a) start out equal and then take independent random walks (say by adding a normally
    distributed increment with mean zero and standard deviation 0.01 to all the q*(a) on each step).
    Prepare plots like Figure 2.2 for an action-value method using sample averages, incrementally computed,
    and another action-value method using a constant step-size parameter, alpha=0.01. Use epsilon=0.1,
    and longer runs, say 10,000 steps.
    """
    bandits = []
    bandits.append(Bandit(epsilon=0.1, sample_averages=True, nonstationary=True))
    bandits.append(Bandit(epsilon=0.1, step_size=0.1, nonstationary=True))
    bandits.append(Bandit(epsilon=0.1, sample_averages=True))
    bandits.append(Bandit(epsilon=0.1, step_size=0.1))
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    plt.plot(rewards[0], label='epsilon = 0.1, sample_averages, non-stationary')
    plt.plot(rewards[1], label='epsilon = 0.1, step_size=0.1, non-stationary')
    plt.plot(rewards[2], label='epsilon = 0.1, sample_averages, stationary')
    plt.plot(rewards[3], label='epsilon = 0.1, step_size=0.1, stationary')
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(best_action_counts[0], label='epsilon = 0.1, sample_averages, non-stationary')
    plt.plot(best_action_counts[1], label='epsilon = 0.1, step_size=0.1, non-stationary')
    plt.plot(best_action_counts[2], label='epsilon = 0.1, sample_averages, stationary')
    plt.plot(best_action_counts[3], label='epsilon = 0.1, step_size=0.1, stationary')
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('../images/exercise_2_5.png')
    plt.show()


def exercise_2_11(runs=2000, time=20000):
    """
    Make a figure analogous to Figure 2.6 for the nonstationary case outlined in Exercise 2.5
    Include the constant-step-size epislon-greedy algorithm with alpha=0.1.
    Use runs of 200,000 steps, and as a performance measure for each algorithm and
    parameter setting, use the average reward over the last 100,000 steps.
    """
    labels = ['epsilon-greedy', 'gradient bandit',
              'UCB', 'optimistic initialization']
    generators = [lambda epsilon: Bandit(epsilon=epsilon, sample_averages=True, nonstationary=True),
                  lambda alpha: Bandit(gradient=True, step_size=alpha, gradient_baseline=True, nonstationary=True),
                  lambda coef: Bandit(epsilon=0, UCB_param=coef, sample_averages=True, nonstationary=True),
                  lambda initial: Bandit(epsilon=0, initial=initial, step_size=0.1, nonstationary=True)]
    parameters = [np.arange(-7, -1, dtype=np.float),
                  np.arange(-5, 2, dtype=np.float),
                  np.arange(-4, 3, dtype=np.float),
                  np.arange(-2, 3, dtype=np.float)]

    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))

    _, average_rewards = simulate(runs, time, bandits)
    print(average_rewards.shape)
    # take the last 10000 steps
    average_rewards = average_rewards[:, 10000:]
    rewards = np.mean(average_rewards, axis=1)

    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i:i + l], label=label)
        i += l
    plt.xlabel('Parameter(2^x)')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('../images/figure_2_6.png')
    plt.close()


if __name__ == "__main__":
    # figure_2_1()
    # figure_2_2()
    # figure_2_3()
    # figure_2_4()
    # figure_2_5()
    # figure_2_6()
    # exercise_2_5()
    exercise_2_11()