# trajectory_sampling.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Trajectory sampling (Chapter 8, Figure 8.8). Compare on-policy trajectory
# sampling against a uniform update distribution on random MDPs.
#
# Input: none. Output: value-of-start-state vs computation.


#######################################################################
# Copyright (C)                                                       #
# 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.use('Agg')

# 2 actions
ACTIONS = [0, 1]

# each transition has a probability to terminate with 0
TERMINATION_PROB = 0.1

# maximum expected updates
MAX_STEPS = 20000

# epsilon greedy for behavior policy
EPSILON = 0.1


# break tie randomly
def argmax(value):
    # TODO: implement argmax
    raise NotImplementedError


class Task:
    # @n_states: number of non-terminal states
    # @b: branch
    # Each episode starts with state 0, and state n_states is a terminal state
    def __init__(self, n_states, b):
        # TODO: implement Task.__init__
        raise NotImplementedError

    def step(self, state, action):
        # TODO: implement Task.step
        raise NotImplementedError


# Evaluate the value of the start state for the greedy policy
# derived from @q under the MDP @task
def evaluate_pi(q, task):
    # use Monte Carlo method to estimate the state value
    # TODO: implement evaluate_pi
    raise NotImplementedError


# perform expected update from a uniform state-action distribution of the MDP @task
# evaluate the learned q value every @eval_interval steps
def uniform(task, eval_interval):
    # TODO: implement uniform
    raise NotImplementedError


# perform expected update from an on-policy distribution of the MDP @task
# evaluate the learned q value every @eval_interval steps
def on_policy(task, eval_interval):
    # TODO: implement on_policy
    raise NotImplementedError


def figure_8_8():
    # TODO: implement figure_8_8
    raise NotImplementedError


if __name__ == '__main__':
    figure_8_8()
