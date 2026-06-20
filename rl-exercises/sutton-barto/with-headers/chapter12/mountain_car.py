# mountain_car.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Mountain Car with eligibility traces (Chapter 12, Figures 12.10 & 12.11).
# Sarsa(lambda) with replacing traces and tile coding.
#
# Input: none. Output: learning curves comparing trace variants.


#######################################################################
# Copyright (C)                                                       #
# 2017-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import floor
from tqdm import tqdm

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

# discount is always 1.0 in these experiments
DISCOUNT = 1.0

# use optimistic initial value, so it's ok to set epsilon to 0
EPSILON = 0

# maximum steps per episode
STEP_LIMIT = 5000

# take an @action at @position and @velocity
# @return: new position, new velocity, reward (always -1)
def step(position, velocity, action):
    # TODO: implement step
    raise NotImplementedError

# accumulating trace update rule
# @trace: old trace (will be modified)
# @activeTiles: current active tile indices
# @lam: lambda
# @return: new trace for convenience
def accumulating_trace(trace, active_tiles, lam):
    # TODO: implement accumulating_trace
    raise NotImplementedError

# replacing trace update rule
# @trace: old trace (will be modified)
# @activeTiles: current active tile indices
# @lam: lambda
# @return: new trace for convenience
def replacing_trace(trace, activeTiles, lam):
    # TODO: implement replacing_trace
    raise NotImplementedError

# replacing trace update rule, 'clearing' means set all tiles corresponding to non-selected actions to 0
# @trace: old trace (will be modified)
# @activeTiles: current active tile indices
# @lam: lambda
# @clearingTiles: tiles to be cleared
# @return: new trace for convenience
def replacing_trace_with_clearing(trace, active_tiles, lam, clearing_tiles):
    # TODO: implement replacing_trace_with_clearing
    raise NotImplementedError

# dutch trace update rule
# @trace: old trace (will be modified)
# @activeTiles: current active tile indices
# @lam: lambda
# @alpha: step size for all tiles
# @return: new trace for convenience
def dutch_trace(trace, active_tiles, lam, alpha):
    # TODO: implement dutch_trace
    raise NotImplementedError

# wrapper class for Sarsa(lambda)
class Sarsa:
    # In this example I use the tiling software instead of implementing standard tiling by myself
    # One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property
    # View the following webpage for more information
    # http://incompleteideas.net/sutton/tiles/tiles3.html
    # @maxSize: the maximum # of indices
    def __init__(self, step_size, lam, trace_update=accumulating_trace, num_of_tilings=8, max_size=2048):
        # TODO: implement Sarsa.__init__
        raise NotImplementedError

    # get indices of active tiles for given state and action
    def get_active_tiles(self, position, velocity, action):
        # I think positionScale * (position - position_min) would be a good normalization.
        # However positionScale * position_min is a constant, so it's ok to ignore it.
        # TODO: implement Sarsa.get_active_tiles
        raise NotImplementedError

    # estimate the value of given state and action
    def value(self, position, velocity, action):
        # TODO: implement Sarsa.value
        raise NotImplementedError

    # learn with given state, action and target
    def learn(self, position, velocity, action, target):
        # TODO: implement Sarsa.learn
        raise NotImplementedError

    # get # of steps to reach the goal under current state value function
    def cost_to_go(self, position, velocity):
        # TODO: implement Sarsa.cost_to_go
        raise NotImplementedError

# get action at @position and @velocity based on epsilon greedy policy and @valueFunction
def get_action(position, velocity, valueFunction):
    # TODO: implement get_action
    raise NotImplementedError

# play Mountain Car for one episode based on given method @evaluator
# @return: total steps in this episode
def play(evaluator):
    # TODO: implement play
    raise NotImplementedError

# figure 12.10, effect of the lambda and alpha on early performance of Sarsa(lambda)
def figure_12_10():
    # TODO: implement figure_12_10
    raise NotImplementedError

# figure 12.11, summary comparision of Sarsa(lambda) algorithms
# I use 8 tilings rather than 10 tilings
def figure_12_11():
    # TODO: implement figure_12_11
    raise NotImplementedError

if __name__ == '__main__':
    figure_12_10()
    figure_12_11()
