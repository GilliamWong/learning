# random_walk.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The 1000-state random walk (Chapter 9, Figures 9.1/9.2/9.5/9.10). On-policy
# prediction with function approximation: gradient Monte Carlo and
# semi-gradient n-step TD using state aggregation, polynomial/Fourier bases,
# and tile coding.
#
# Input: none. Output: approximate value functions and error curves.


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

# # of states except for terminal states
N_STATES = 1000

# all states
STATES = np.arange(1, N_STATES + 1)

# start from a central state
START_STATE = 500

# terminal states
END_STATES = [0, N_STATES + 1]

# possible actions
ACTION_LEFT = -1
ACTION_RIGHT = 1
ACTIONS = [ACTION_LEFT, ACTION_RIGHT]

# maximum stride for an action
STEP_RANGE = 100

def compute_true_value():
    # true state value, just a promising guess
    # TODO: implement compute_true_value
    raise NotImplementedError

# take an @action at @state, return new state and reward for this transition
def step(state, action):
    # TODO: implement step
    raise NotImplementedError

# get an action, following random policy
def get_action():
    # TODO: implement get_action
    raise NotImplementedError

# a wrapper class for aggregation value function
class ValueFunction:
    # @num_of_groups: # of aggregations
    def __init__(self, num_of_groups):
        # TODO: implement ValueFunction.__init__
        raise NotImplementedError

    # get the value of @state
    def value(self, state):
        # TODO: implement ValueFunction.value
        raise NotImplementedError

    # update parameters
    # @delta: step size * (target - old estimation)
    # @state: state of current sample
    def update(self, delta, state):
        # TODO: implement ValueFunction.update
        raise NotImplementedError

# a wrapper class for tile coding value function
class TilingsValueFunction:
    # @num_of_tilings: # of tilings
    # @tileWidth: each tiling has several tiles, this parameter specifies the width of each tile
    # @tilingOffset: specifies how tilings are put together
    def __init__(self, numOfTilings, tileWidth, tilingOffset):
        # TODO: implement TilingsValueFunction.__init__
        raise NotImplementedError

    # get the value of @state
    def value(self, state):
        # TODO: implement TilingsValueFunction.value
        raise NotImplementedError

    # update parameters
    # @delta: step size * (target - old estimation)
    # @state: state of current sample
    def update(self, delta, state):

        # each state is covered by same number of tilings
        # so the delta should be divided equally into each tiling (tile)
        # TODO: implement TilingsValueFunction.update
        raise NotImplementedError

# a wrapper class for polynomial / Fourier -based value function
POLYNOMIAL_BASES = 0
FOURIER_BASES = 1
class BasesValueFunction:
    # @order: # of bases, each function also has one more constant parameter (called bias in machine learning)
    # @type: polynomial bases or Fourier bases
    def __init__(self, order, type):
        # TODO: implement BasesValueFunction.__init__
        raise NotImplementedError

    # get the value of @state
    def value(self, state):
        # map the state space into [0, 1]
        # TODO: implement BasesValueFunction.value
        raise NotImplementedError

    def update(self, delta, state):
        # map the state space into [0, 1]
        # TODO: implement BasesValueFunction.update
        raise NotImplementedError

# gradient Monte Carlo algorithm
# @value_function: an instance of class ValueFunction
# @alpha: step size
# @distribution: array to store the distribution statistics
def gradient_monte_carlo(value_function, alpha, distribution=None):
    # TODO: implement gradient_monte_carlo
    raise NotImplementedError

# semi-gradient n-step TD algorithm
# @valueFunction: an instance of class ValueFunction
# @n: # of steps
# @alpha: step size
def semi_gradient_temporal_difference(value_function, n, alpha):
    # initial starting state
    # TODO: implement semi_gradient_temporal_difference
    raise NotImplementedError

# Figure 9.1, gradient Monte Carlo algorithm
def figure_9_1(true_value):
    # TODO: implement figure_9_1
    raise NotImplementedError

# semi-gradient TD on 1000-state random walk
def figure_9_2_left(true_value):
    # TODO: implement figure_9_2_left
    raise NotImplementedError

# different alphas and steps for semi-gradient TD
def figure_9_2_right(true_value):
    # all possible steps
    # TODO: implement figure_9_2_right
    raise NotImplementedError

def figure_9_2(true_value):
    # TODO: implement figure_9_2
    raise NotImplementedError

# Figure 9.5, Fourier basis and polynomials
def figure_9_5(true_value):
    # my machine can only afford 1 run
    # TODO: implement figure_9_5
    raise NotImplementedError

# Figure 9.10, it will take quite a while
def figure_9_10(true_value):

    # My machine can only afford one run, thus the curve isn't so smooth
    # TODO: implement figure_9_10
    raise NotImplementedError

if __name__ == '__main__':
    true_value = compute_true_value()

    figure_9_1(true_value)
    figure_9_2(true_value)
    figure_9_5(true_value)
    figure_9_10(true_value)
