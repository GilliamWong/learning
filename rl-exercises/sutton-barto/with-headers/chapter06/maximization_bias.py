# maximization_bias.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Maximization bias (Chapter 6, Figure 6.7). Compare Q-learning and Double
# Q-learning on the small MDP that exposes maximization bias.
#
# Input: none. Output: %-left-action vs episode curves.


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
import copy

# state A
STATE_A = 0

# state B
STATE_B = 1

# use one terminal state
STATE_TERMINAL = 2

# starts from state A
STATE_START = STATE_A

# possible actions in A
ACTION_A_RIGHT = 0
ACTION_A_LEFT = 1

# probability for exploration
EPSILON = 0.1

# step size
ALPHA = 0.1

# discount for max value
GAMMA = 1.0

# possible actions in B, maybe 10 actions
ACTIONS_B = range(0, 10)

# all possible actions
STATE_ACTIONS = [[ACTION_A_RIGHT, ACTION_A_LEFT], ACTIONS_B]

# state action pair values, if a state is a terminal state, then the value is always 0
INITIAL_Q = [np.zeros(2), np.zeros(len(ACTIONS_B)), np.zeros(1)]

# set up destination for each state and each action
TRANSITION = [[STATE_TERMINAL, STATE_B], [STATE_TERMINAL] * len(ACTIONS_B)]

# choose an action based on epsilon greedy algorithm
def choose_action(state, q_value):
    # TODO: implement choose_action
    raise NotImplementedError

# take @action in @state, return the reward
def take_action(state, action):
    # TODO: implement take_action
    raise NotImplementedError

# if there are two state action pair value array, use double Q-Learning
# otherwise use normal Q-Learning
def q_learning(q1, q2=None):
    # TODO: implement q_learning
    raise NotImplementedError

# Figure 6.7, 1,000 runs may be enough, # of actions in state B will also affect the curves
def figure_6_7():
    # each independent run has 300 episodes
    # TODO: implement figure_6_7
    raise NotImplementedError

if __name__ == '__main__':
    figure_6_7()