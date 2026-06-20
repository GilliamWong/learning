# short_corridor.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The short-corridor gridworld (Chapter 13, Figures 13.1 & 13.2).
# Policy-gradient control with REINFORCE (and REINFORCE with baseline) on a
# small problem whose optimal policy is stochastic.
#
# Input: none. Output: total-reward-vs-episode curves.


#######################################################################
# Copyright (C)                                                       #
# 2018 Sergii Bondariev (sergeybondarev@gmail.com)                    #
# 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

def true_value(p):
    """ True value of the first state
    Args:
        p (float): probability of the action 'right'.
    Returns:
        True value of the first state.
        The expression is obtained by manually solving the easy linear system
        of Bellman equations using known dynamics.
    """
    # TODO: implement true_value
    raise NotImplementedError

class ShortCorridor:
    """
    Short corridor environment, see Example 13.1
    """
    def __init__(self):
        # TODO: implement ShortCorridor.__init__
        raise NotImplementedError

    def reset(self):
        # TODO: implement ShortCorridor.reset
        raise NotImplementedError

    def step(self, go_right):
        """
        Args:
            go_right (bool): chosen action
        Returns:
            tuple of (reward, episode terminated?)
        """
        # TODO: implement ShortCorridor.step
        raise NotImplementedError

def softmax(x):
    # TODO: implement softmax
    raise NotImplementedError

class ReinforceAgent:
    """
    ReinforceAgent that follows algorithm
    'REINFORNCE Monte-Carlo Policy-Gradient Control (episodic)'
    """
    def __init__(self, alpha, gamma):
        # set values such that initial conditions correspond to left-epsilon greedy
        # TODO: implement ReinforceAgent.__init__
        raise NotImplementedError

    def get_pi(self):
        # TODO: implement ReinforceAgent.get_pi
        raise NotImplementedError

    def get_p_right(self):
        # TODO: implement ReinforceAgent.get_p_right
        raise NotImplementedError

    def choose_action(self, reward):
        # TODO: implement ReinforceAgent.choose_action
        raise NotImplementedError

    def episode_end(self, last_reward):
        # TODO: implement ReinforceAgent.episode_end
        raise NotImplementedError

class ReinforceBaselineAgent(ReinforceAgent):
    def __init__(self, alpha, gamma, alpha_w):
        # TODO: implement ReinforceBaselineAgent.__init__
        raise NotImplementedError

    def episode_end(self, last_reward):
        # TODO: implement ReinforceBaselineAgent.episode_end
        raise NotImplementedError

def trial(num_episodes, agent_generator):
    # TODO: implement trial
    raise NotImplementedError

def example_13_1():
    # TODO: implement example_13_1
    raise NotImplementedError

def figure_13_1():
    # TODO: implement figure_13_1
    raise NotImplementedError

def figure_13_2():
    # TODO: implement figure_13_2
    raise NotImplementedError

if __name__ == '__main__':
    example_13_1()
    figure_13_1()
    figure_13_2()
