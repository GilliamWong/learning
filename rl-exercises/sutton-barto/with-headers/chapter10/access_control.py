# access_control.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The access-control queuing task (Chapter 10, Figure 10.5). Average-reward
# control with differential semi-gradient Sarsa.
#
# Input: none. Output: the learned value/policy over free servers and
# priorities.


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
import seaborn as sns

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

# possible priorities
PRIORITIES = np.arange(0, 4)
# reward for each priority
REWARDS = np.power(2, np.arange(0, 4))

# possible actions
REJECT = 0
ACCEPT = 1
ACTIONS = [REJECT, ACCEPT]

# total number of servers
NUM_OF_SERVERS = 10

# at each time step, a busy server will be free w.p. 0.06
PROBABILITY_FREE = 0.06

# step size for learning state-action value
ALPHA = 0.01

# step size for learning average reward
BETA = 0.01

# probability for exploration
EPSILON = 0.1

# a wrapper class for differential semi-gradient Sarsa state-action function
class ValueFunction:
    # In this example I use the tiling software instead of implementing standard tiling by myself
    # One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property
    # View the following webpage for more information
    # http://incompleteideas.net/sutton/tiles/tiles3.html
    # @alpha: step size for learning state-action value
    # @beta: step size for learning average reward
    def __init__(self, num_of_tilings, alpha=ALPHA, beta=BETA):
        # TODO: implement ValueFunction.__init__
        raise NotImplementedError

    # get indices of active tiles for given state and action
    def get_active_tiles(self, free_servers, priority, action):
        # TODO: implement ValueFunction.get_active_tiles
        raise NotImplementedError

    # estimate the value of given state and action without subtracting average
    def value(self, free_servers, priority, action):
        # TODO: implement ValueFunction.value
        raise NotImplementedError

    # estimate the value of given state without subtracting average
    def state_value(self, free_servers, priority):
        # TODO: implement ValueFunction.state_value
        raise NotImplementedError

    # learn with given sequence
    def learn(self, free_servers, priority, action, new_free_servers, new_priority, new_action, reward):
        # TODO: implement ValueFunction.learn
        raise NotImplementedError

# get action based on epsilon greedy policy and @valueFunction
def get_action(free_servers, priority, value_function):
    # if no free server, can't accept
    # TODO: implement get_action
    raise NotImplementedError

# take an action
def take_action(free_servers, priority, action):
    # TODO: implement take_action
    raise NotImplementedError

# differential semi-gradient Sarsa
# @valueFunction: state value function to learn
# @maxSteps: step limit in the continuing task
def differential_semi_gradient_sarsa(value_function, max_steps):
    # TODO: implement differential_semi_gradient_sarsa
    raise NotImplementedError

# Figure 10.5, Differential semi-gradient Sarsa on the access-control queuing task
def figure_10_5():
    # TODO: implement figure_10_5
    raise NotImplementedError

if __name__ == '__main__':
    figure_10_5()
