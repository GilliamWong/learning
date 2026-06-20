# counterexample.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Baird's counterexample (Chapter 11, Figures 11.2/11.6/11.7). Demonstrate
# off-policy divergence of semi-gradient TD, and the stabler behavior of
# TDC/Gradient-TD and Emphatic-TD.
#
# Input: none. Output: weight-trajectory curves.


#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
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

# all states: state 0-5 are upper states
STATES = np.arange(0, 7)
# state 6 is lower state
LOWER_STATE = 6
# discount factor
DISCOUNT = 0.99

# each state is represented by a vector of length 8
FEATURE_SIZE = 8
FEATURES = np.zeros((len(STATES), FEATURE_SIZE))
for i in range(LOWER_STATE):
    FEATURES[i, i] = 2
    FEATURES[i, 7] = 1
FEATURES[LOWER_STATE, 6] = 1
FEATURES[LOWER_STATE, 7] = 2

# all possible actions
DASHED = 0
SOLID = 1
ACTIONS = [DASHED, SOLID]

# reward is always zero
REWARD = 0

# take @action at @state, return the new state
def step(state, action):
    # TODO: implement step
    raise NotImplementedError

# target policy
def target_policy(state):
    # TODO: implement target_policy
    raise NotImplementedError

# state distribution for the behavior policy
STATE_DISTRIBUTION = np.ones(len(STATES)) / 7
STATE_DISTRIBUTION_MAT = np.matrix(np.diag(STATE_DISTRIBUTION))
# projection matrix for minimize MSVE
PROJECTION_MAT = np.matrix(FEATURES) * \
                 np.linalg.pinv(np.matrix(FEATURES.T) * STATE_DISTRIBUTION_MAT * np.matrix(FEATURES)) * \
                 np.matrix(FEATURES.T) * \
                 STATE_DISTRIBUTION_MAT

# behavior policy
BEHAVIOR_SOLID_PROBABILITY = 1.0 / 7
def behavior_policy(state):
    # TODO: implement behavior_policy
    raise NotImplementedError

# Semi-gradient off-policy temporal difference
# @state: current state
# @theta: weight for each component of the feature vector
# @alpha: step size
# @return: next state
def semi_gradient_off_policy_TD(state, theta, alpha):
    # TODO: implement semi_gradient_off_policy_TD
    raise NotImplementedError

# Semi-gradient DP
# @theta: weight for each component of the feature vector
# @alpha: step size
def semi_gradient_DP(theta, alpha):
    # TODO: implement semi_gradient_DP
    raise NotImplementedError

# temporal difference with gradient correction
# @state: current state
# @theta: weight of each component of the feature vector
# @weight: auxiliary trace for gradient correction
# @alpha: step size of @theta
# @beta: step size of @weight
def TDC(state, theta, weight, alpha, beta):
    # TODO: implement TDC
    raise NotImplementedError

# expected temporal difference with gradient correction
# @theta: weight of each component of the feature vector
# @weight: auxiliary trace for gradient correction
# @alpha: step size of @theta
# @beta: step size of @weight
def expected_TDC(theta, weight, alpha, beta):
    # TODO: implement expected_TDC
    raise NotImplementedError

    # if *accumulate* expected update and actually apply update here, then it's synchronous
    # theta += alpha * expectedUpdateTheta
    # weight += beta * expectedUpdateWeight

# interest is 1 for every state
INTEREST = 1

# expected update of ETD
# @theta: weight of each component of the feature vector
# @emphasis: current emphasis
# @alpha: step size of @theta
# @return: expected next emphasis
def expected_emphatic_TD(theta, emphasis, alpha):
    # we perform synchronous update for both theta and emphasis
    # TODO: implement expected_emphatic_TD
    raise NotImplementedError

# compute RMSVE for a value function parameterized by @theta
# true value function is always 0 in this example
def compute_RMSVE(theta):
    # TODO: implement compute_RMSVE
    raise NotImplementedError

# compute RMSPBE for a value function parameterized by @theta
# true value function is always 0 in this example
def compute_RMSPBE(theta):
    # TODO: implement compute_RMSPBE
    raise NotImplementedError

figureIndex = 0

# Figure 11.2(left), semi-gradient off-policy TD
def figure_11_2_left():
    # Initialize the theta
    # TODO: implement figure_11_2_left
    raise NotImplementedError

# Figure 11.2(right), semi-gradient DP
def figure_11_2_right():
    # Initialize the theta
    # TODO: implement figure_11_2_right
    raise NotImplementedError

def figure_11_2():
    # TODO: implement figure_11_2
    raise NotImplementedError

# Figure 11.6(left), temporal difference with gradient correction
def figure_11_6_left():
    # Initialize the theta
    # TODO: implement figure_11_6_left
    raise NotImplementedError

# Figure 11.6(right), expected temporal difference with gradient correction
def figure_11_6_right():
    # Initialize the theta
    # TODO: implement figure_11_6_right
    raise NotImplementedError

def figure_11_6():
    # TODO: implement figure_11_6
    raise NotImplementedError

# Figure 11.7, expected ETD
def figure_11_7():
    # Initialize the theta
    # TODO: implement figure_11_7
    raise NotImplementedError

if __name__ == '__main__':
    figure_11_2()
    figure_11_6()
    figure_11_7()
