# gamblers_problem.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The gambler's problem (Chapter 4, Figure 4.3). Value iteration on a
# coin-flip betting MDP.
#
# Input: none. Output: value estimates across sweeps and the final
# capital->stake policy.


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

matplotlib.use('Agg')

# goal
GOAL = 100

# all states, including state 0 and state 100
STATES = np.arange(GOAL + 1)

# probability of head
HEAD_PROB = 0.4


def figure_4_3():
    # state value
    # TODO: implement figure_4_3
    raise NotImplementedError


if __name__ == '__main__':
    figure_4_3()
