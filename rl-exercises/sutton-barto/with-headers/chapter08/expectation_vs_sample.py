# expectation_vs_sample.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Expected vs sample updates (Chapter 8, Figure 8.7). Compare the efficiency
# of expected and sample updates as a function of branching factor.
#
# Input: none. Output: error-vs-computation curves.


#######################################################################
# Copyright (C)                                                       #
# 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# for figure 8.7, run a simulation of 2 * @b steps
def b_steps(b):
    # set the value of the next b states
    # it is not clear how to set this
    # TODO: implement b_steps
    raise NotImplementedError

def figure_8_7():
    # TODO: implement figure_8_7
    raise NotImplementedError

if __name__ == '__main__':
    figure_8_7()
