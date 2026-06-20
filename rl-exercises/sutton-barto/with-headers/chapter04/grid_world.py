# grid_world.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The 4x4 gridworld (Chapter 4, Figure 4.1). Iterative policy evaluation of
# the equiprobable random policy until convergence.
#
# Input: none. Output: the value function across sweeps.


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

WORLD_SIZE = 4
# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROB = 0.25


def is_terminal(state):
    # TODO: implement is_terminal
    raise NotImplementedError


def step(state, action):
    # TODO: implement step
    raise NotImplementedError


def draw_image(image):
    # TODO: implement draw_image
    raise NotImplementedError


def compute_state_value(in_place=True, discount=1.0):
    # TODO: implement compute_state_value
    raise NotImplementedError


def figure_4_1():
    # While the author suggests using in-place iterative policy evaluation,
    # Figure 4.1 actually uses out-of-place version.
    # TODO: implement figure_4_1
    raise NotImplementedError


if __name__ == '__main__':
    figure_4_1()
