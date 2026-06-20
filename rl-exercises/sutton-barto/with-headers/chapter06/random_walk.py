# random_walk.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The 5-state random walk (Chapter 6, Example 6.2 / Figure 6.2). Compare TD(0)
# and constant-alpha Monte Carlo value estimation, including batch updating.
#
# Input: none. Output: learned values and RMS-error curves.


#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# 0 is the left terminal state
# 6 is the right terminal state
# 1 ... 5 represents A ... E
VALUES = np.zeros(7)
VALUES[1:6] = 0.5
# For convenience, we assume all rewards are 0
# and the left terminal state has value 0, the right terminal state has value 1
# This trick has been used in Gambler's Problem
VALUES[6] = 1

# set up true state values
TRUE_VALUE = np.zeros(7)
TRUE_VALUE[1:6] = np.arange(1, 6) / 6.0
TRUE_VALUE[6] = 1

ACTION_LEFT = 0
ACTION_RIGHT = 1

# @values: current states value, will be updated if @batch is False
# @alpha: step size
# @batch: whether to update @values
def temporal_difference(values, alpha=0.1, batch=False):
    # TODO: implement temporal_difference
    raise NotImplementedError

# @values: current states value, will be updated if @batch is False
# @alpha: step size
# @batch: whether to update @values
def monte_carlo(values, alpha=0.1, batch=False):
    # TODO: implement monte_carlo
    raise NotImplementedError

# Example 6.2 left
def compute_state_value():
    # TODO: implement compute_state_value
    raise NotImplementedError

# Example 6.2 right
def rms_error():
    # Same alpha value can appear in both arrays
    # TODO: implement rms_error
    raise NotImplementedError

# Figure 6.2
# @method: 'TD' or 'MC'
def batch_updating(method, episodes, alpha=0.001):
    # perform 100 independent runs
    # TODO: implement batch_updating
    raise NotImplementedError

def example_6_2():
    # TODO: implement example_6_2
    raise NotImplementedError

def figure_6_2():
    # TODO: implement figure_6_2
    raise NotImplementedError

if __name__ == '__main__':
    example_6_2()
    figure_6_2()
