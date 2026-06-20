# grid_world.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The 5x5 gridworld (Chapter 3, Figures 3.2 & 3.5). Compute the state-value
# function under the equiprobable random policy and the optimal value
# function/policy by solving the Bellman expectation and optimality equations.
#
# Input: none. Output: value-grid figures.


#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use('Agg')

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
DISCOUNT = 0.9

# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTIONS_FIGS=[ '←', '↑', '→', '↓']


ACTION_PROB = 0.25


def step(state, action):
    # TODO: implement step
    raise NotImplementedError


def draw_image(image):
    # TODO: implement draw_image
    raise NotImplementedError

def draw_policy(optimal_values):
    # TODO: implement draw_policy
    raise NotImplementedError


def figure_3_2():
    # TODO: implement figure_3_2
    raise NotImplementedError

def figure_3_2_linear_system():
    '''
    Here we solve the linear system of equations to find the exact solution.
    We do this by filling the coefficients for each of the states with their respective right side constant.
    '''
    # TODO: implement figure_3_2_linear_system
    raise NotImplementedError

def figure_3_5():
    # TODO: implement figure_3_5
    raise NotImplementedError


if __name__ == '__main__':
    figure_3_2_linear_system()
    figure_3_2()
    figure_3_5()
