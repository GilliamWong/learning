# ten_armed_testbed.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The 10-armed bandit testbed (Chapter 2, Figures 2.1-2.6). Implement a
# k-armed bandit and the action-value methods that learn on it:
# epsilon-greedy, optimistic initial values, UCB, and gradient bandits.
#
# Input: none (parameters are module constants). Output: matplotlib figures of
# average reward and % optimal action.


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
from tqdm import trange

matplotlib.use('Agg')


class Bandit:
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, true_reward=0.):
        # TODO: implement Bandit.__init__
        raise NotImplementedError

    def reset(self):
        # real reward for each action
        # TODO: implement Bandit.reset
        raise NotImplementedError

    # get an action for this bandit
    def act(self):
        # TODO: implement Bandit.act
        raise NotImplementedError

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        # TODO: implement Bandit.step
        raise NotImplementedError


def simulate(runs, time, bandits):
    # TODO: implement simulate
    raise NotImplementedError


def figure_2_1():
    # TODO: implement figure_2_1
    raise NotImplementedError


def figure_2_2(runs=2000, time=1000):
    # TODO: implement figure_2_2
    raise NotImplementedError


def figure_2_3(runs=2000, time=1000):
    # TODO: implement figure_2_3
    raise NotImplementedError


def figure_2_4(runs=2000, time=1000):
    # TODO: implement figure_2_4
    raise NotImplementedError


def figure_2_5(runs=2000, time=1000):
    # TODO: implement figure_2_5
    raise NotImplementedError


def figure_2_6(runs=2000, time=1000):
    # TODO: implement figure_2_6
    raise NotImplementedError


if __name__ == '__main__':
    figure_2_1()
    figure_2_2()
    figure_2_3()
    figure_2_4()
    figure_2_5()
    figure_2_6()
