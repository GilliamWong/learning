# infinite_variance.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The infinite-variance example (Chapter 5, Figure 5.4). Show that ordinary
# importance-sampling estimates can have infinite variance on a one-state
# problem.
#
# Input: none. Output: estimate trajectories across many runs.


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

ACTION_BACK = 0
ACTION_END = 1

# behavior policy
def behavior_policy():
    # TODO: implement behavior_policy
    raise NotImplementedError

# target policy
def target_policy():
    # TODO: implement target_policy
    raise NotImplementedError

# one turn
def play():
    # track the action for importance ratio
    # TODO: implement play
    raise NotImplementedError

def figure_5_4():
    # TODO: implement figure_5_4
    raise NotImplementedError

if __name__ == '__main__':
    figure_5_4()
