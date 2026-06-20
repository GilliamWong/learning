# random_walk.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The 19-state random walk with eligibility traces (Chapter 12, Figures
# 12.3/12.6/12.8). Implement the offline lambda-return algorithm, TD(lambda),
# and true online TD(lambda).
#
# Input: none. Output: RMS-error vs (alpha, lambda) curves.


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

# all states
N_STATES = 19

# all states but terminal states
STATES = np.arange(1, N_STATES + 1)

# start from the middle state
START_STATE = 10

# two terminal states
# an action leading to the left terminal state has reward -1
# an action leading to the right terminal state has reward 1
END_STATES = [0, N_STATES + 1]

# true state values from Bellman equation
TRUE_VALUE = np.arange(-20, 22, 2) / 20.0
TRUE_VALUE[0] = TRUE_VALUE[N_STATES + 1] = 0.0

# base class for lambda-based algorithms in this chapter
# In this example, we use the simplest linear feature function, state aggregation.
# And we use exact 19 groups, so the weights for each group is exact the value for that state
class ValueFunction:
    # @rate: lambda, as it's a keyword in python, so I call it rate
    # @stepSize: alpha, step size for update
    def __init__(self, rate, step_size):
        # TODO: implement ValueFunction.__init__
        raise NotImplementedError

    # the state value is just the weight
    def value(self, state):
        # TODO: implement ValueFunction.value
        raise NotImplementedError

    # feed the algorithm with new observation
    # derived class should override this function
    def learn(self, state, reward):
        # TODO: implement ValueFunction.learn
        raise NotImplementedError

    # initialize some variables at the beginning of each episode
    # must be called at the very beginning of each episode
    # derived class should override this function
    def new_episode(self):
        # TODO: implement ValueFunction.new_episode
        raise NotImplementedError

# Off-line lambda-return algorithm
class OffLineLambdaReturn(ValueFunction):
    def __init__(self, rate, step_size):
        # TODO: implement OffLineLambdaReturn.__init__
        raise NotImplementedError

    def new_episode(self):
        # initialize the trajectory
        # TODO: implement OffLineLambdaReturn.new_episode
        raise NotImplementedError

    def learn(self, state, reward):
        # add the new state to the trajectory
        # TODO: implement OffLineLambdaReturn.learn
        raise NotImplementedError

    # get the n-step return from the given time
    def n_step_return_from_time(self, n, time):
        # gamma is always 1 and rewards are zero except for the last reward
        # the formula can be simplified
        # TODO: implement OffLineLambdaReturn.n_step_return_from_time
        raise NotImplementedError

    # get the lambda-return from the given time
    def lambda_return_from_time(self, time):
        # TODO: implement OffLineLambdaReturn.lambda_return_from_time
        raise NotImplementedError

    # perform off-line learning at the end of an episode
    def off_line_learn(self):
        # TODO: implement OffLineLambdaReturn.off_line_learn
        raise NotImplementedError

# TD(lambda) algorithm
class TemporalDifferenceLambda(ValueFunction):
    def __init__(self, rate, step_size):
        # TODO: implement TemporalDifferenceLambda.__init__
        raise NotImplementedError

    def new_episode(self):
        # initialize the eligibility trace
        # TODO: implement TemporalDifferenceLambda.new_episode
        raise NotImplementedError

    def learn(self, state, reward):
        # update the eligibility trace and weights
        # TODO: implement TemporalDifferenceLambda.learn
        raise NotImplementedError

# True online TD(lambda) algorithm
class TrueOnlineTemporalDifferenceLambda(ValueFunction):
    def __init__(self, rate, step_size):
        # TODO: implement TrueOnlineTemporalDifferenceLambda.__init__
        raise NotImplementedError

    def new_episode(self):
        # initialize the eligibility trace
        # TODO: implement TrueOnlineTemporalDifferenceLambda.new_episode
        raise NotImplementedError

    def learn(self, state, reward):
        # update the eligibility trace and weights
        # TODO: implement TrueOnlineTemporalDifferenceLambda.learn
        raise NotImplementedError

# 19-state random walk
def random_walk(value_function):
    # TODO: implement random_walk
    raise NotImplementedError

# general plot framework
# @valueFunctionGenerator: generate an instance of value function
# @runs: specify the number of independent runs
# @lambdas: a series of different lambda values
# @alphas: sequences of step size for each lambda
def parameter_sweep(value_function_generator, runs, lambdas, alphas):
    # play for 10 episodes for each run
    # TODO: implement parameter_sweep
    raise NotImplementedError

# Figure 12.3: Off-line lambda-return algorithm
def figure_12_3():
    # TODO: implement figure_12_3
    raise NotImplementedError

# Figure 12.6: TD(lambda) algorithm
def figure_12_6():
    # TODO: implement figure_12_6
    raise NotImplementedError

# Figure 12.7: True online TD(lambda) algorithm
def figure_12_8():
    # TODO: implement figure_12_8
    raise NotImplementedError

if __name__ == '__main__':
    figure_12_3()
    figure_12_6()
    figure_12_8()
