# mountain_car.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Mountain Car (Chapter 10, Figures 10.1-10.4). On-policy control with
# function approximation: episodic semi-gradient Sarsa with tile coding (and
# n-step variants).
#
# Input: none. Output: cost-to-go surfaces and learning curves.


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
from mpl_toolkits.mplot3d.axes3d import Axes3D
from math import floor

#######################################################################
# Following are some utilities for tile coding from Rich.
# To make each file self-contained, I copied them from
# http://incompleteideas.net/tiles/tiles3.py-remove
# with some naming convention changes
#
# Tile coding starts
class IHT:
    "Structure to handle collisions"
    def __init__(self, size_val):
        # TODO: implement IHT.__init__
        raise NotImplementedError

    def count(self):
        # TODO: implement IHT.count
        raise NotImplementedError

    def full(self):
        # TODO: implement IHT.full
        raise NotImplementedError

    def get_index(self, obj, read_only=False):
        # TODO: implement IHT.get_index
        raise NotImplementedError

def hash_coords(coordinates, m, read_only=False):
    # TODO: implement hash_coords
    raise NotImplementedError

def tiles(iht_or_size, num_tilings, floats, ints=None, read_only=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    # TODO: implement tiles
    raise NotImplementedError
# Tile coding ends
#######################################################################

# all possible actions
ACTION_REVERSE = -1
ACTION_ZERO = 0
ACTION_FORWARD = 1
# order is important
ACTIONS = [ACTION_REVERSE, ACTION_ZERO, ACTION_FORWARD]

# bound for position and velocity
POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07

# use optimistic initial value, so it's ok to set epsilon to 0
EPSILON = 0

# take an @action at @position and @velocity
# @return: new position, new velocity, reward (always -1)
def step(position, velocity, action):
    # TODO: implement step
    raise NotImplementedError

# wrapper class for state action value function
class ValueFunction:
    # In this example I use the tiling software instead of implementing standard tiling by myself
    # One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property
    # View the following webpage for more information
    # http://incompleteideas.net/sutton/tiles/tiles3.html
    # @max_size: the maximum # of indices
    def __init__(self, step_size, num_of_tilings=8, max_size=2048):
        # TODO: implement ValueFunction.__init__
        raise NotImplementedError

    # get indices of active tiles for given state and action
    def get_active_tiles(self, position, velocity, action):
        # I think positionScale * (position - position_min) would be a good normalization.
        # However positionScale * position_min is a constant, so it's ok to ignore it.
        # TODO: implement ValueFunction.get_active_tiles
        raise NotImplementedError

    # estimate the value of given state and action
    def value(self, position, velocity, action):
        # TODO: implement ValueFunction.value
        raise NotImplementedError

    # learn with given state, action and target
    def learn(self, position, velocity, action, target):
        # TODO: implement ValueFunction.learn
        raise NotImplementedError

    # get # of steps to reach the goal under current state value function
    def cost_to_go(self, position, velocity):
        # TODO: implement ValueFunction.cost_to_go
        raise NotImplementedError

# get action at @position and @velocity based on epsilon greedy policy and @valueFunction
def get_action(position, velocity, value_function):
    # TODO: implement get_action
    raise NotImplementedError

# semi-gradient n-step Sarsa
# @valueFunction: state value function to learn
# @n: # of steps
def semi_gradient_n_step_sarsa(value_function, n=1):
    # start at a random position around the bottom of the valley
    # TODO: implement semi_gradient_n_step_sarsa
    raise NotImplementedError

# print learned cost to go
def print_cost(value_function, episode, ax):
    # TODO: implement print_cost
    raise NotImplementedError

# Figure 10.1, cost to go in a single run
def figure_10_1():
    # TODO: implement figure_10_1
    raise NotImplementedError

# Figure 10.2, semi-gradient Sarsa with different alphas
def figure_10_2():
    # TODO: implement figure_10_2
    raise NotImplementedError

# Figure 10.3, one-step semi-gradient Sarsa vs multi-step semi-gradient Sarsa
def figure_10_3():
    # TODO: implement figure_10_3
    raise NotImplementedError

# Figure 10.4, effect of alpha and n on multi-step semi-gradient Sarsa
def figure_10_4():
    # TODO: implement figure_10_4
    raise NotImplementedError

if __name__ == '__main__':
    figure_10_1()
    figure_10_2()
    figure_10_3()
    figure_10_4()
