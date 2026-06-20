# windy_grid_world.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The windy gridworld (Chapter 6, Figure 6.3 / Example 6.5). On-policy TD
# control with Sarsa in a gridworld with upward 'wind'.
#
# Input: none. Output: the learning curve (time steps vs episodes).


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

# world height
WORLD_HEIGHT = 7

# world width
WORLD_WIDTH = 10

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# probability for exploration
EPSILON = 0.1

# Sarsa step size
ALPHA = 0.5

# reward for each step
REWARD = -1.0

START = [3, 0]
GOAL = [3, 7]
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

def step(state, action):
    # TODO: implement step
    raise NotImplementedError

# play for an episode
def episode(q_value):
    # track the total time steps in this episode
    # TODO: implement episode
    raise NotImplementedError

def figure_6_3():
    # TODO: implement figure_6_3
    raise NotImplementedError

if __name__ == '__main__':
    figure_6_3()

