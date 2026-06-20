# car_rental_synchronous.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Jack's car rental (Chapter 4, Figure 4.2), synchronous-update variant. Same
# policy-iteration problem with synchronous value sweeps.
#
# Input: none. Output: improved policies and the final value function.


#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Aja Rangaswamy (aja004@gmail.com)                              #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# This file is contributed by Tahsincan Köse which implements a synchronous policy evaluation, while the car_rental.py
# implements an asynchronous policy evaluation. This file also utilizes multi-processing for acceleration and contains
# an answer to Exercise 4.5

import numpy as np
import matplotlib.pyplot as plt
import math
import tqdm
import multiprocessing as mp
from functools import partial
import time
import itertools

############# PROBLEM SPECIFIC CONSTANTS #######################
MAX_CARS = 20
MAX_MOVE = 5
MOVE_COST = -2
ADDITIONAL_PARK_COST = -4

RENT_REWARD = 10
# expectation for rental requests in first location
RENTAL_REQUEST_FIRST_LOC = 3
# expectation for rental requests in second location
RENTAL_REQUEST_SECOND_LOC = 4
# expectation for # of cars returned in first location
RETURNS_FIRST_LOC = 3
# expectation for # of cars returned in second location
RETURNS_SECOND_LOC = 2
################################################################

poisson_cache = dict()


def poisson(n, lam):
    # TODO: implement poisson
    raise NotImplementedError


class PolicyIteration:
    def __init__(self, truncate, parallel_processes, delta=1e-2, gamma=0.9, solve_4_5=False):
        # TODO: implement PolicyIteration.__init__
        raise NotImplementedError

    def solve(self):
        # TODO: implement PolicyIteration.solve
        raise NotImplementedError

    # out-place
    def policy_evaluation(self, values, policy):

        # TODO: implement PolicyIteration.policy_evaluation
        raise NotImplementedError

    def policy_improvement(self, actions, values, policy):
        # TODO: implement PolicyIteration.policy_improvement
        raise NotImplementedError

    # O(n^4) computation for all possible requests and returns
    def bellman(self, values, action, state):
        # TODO: implement PolicyIteration.bellman
        raise NotImplementedError

    # Parallelization enforced different helper functions
    # Expected return calculator for Policy Evaluation
    def expected_return_pe(self, policy, values, state):

        # TODO: implement PolicyIteration.expected_return_pe
        raise NotImplementedError

    # Expected return calculator for Policy Improvement
    def expected_return_pi(self, values, action, state):

        # TODO: implement PolicyIteration.expected_return_pi
        raise NotImplementedError

    def plot(self):
        # TODO: implement PolicyIteration.plot
        raise NotImplementedError


if __name__ == '__main__':
    TRUNCATE = 9
    solver = PolicyIteration(TRUNCATE, parallel_processes=4, delta=1e-1, gamma=0.9, solve_4_5=True)
    solver.solve()
    solver.plot()
