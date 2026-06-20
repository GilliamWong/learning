# blackjack.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Blackjack via Monte Carlo (Chapter 5, Figures 5.1-5.3). Implement Monte
# Carlo prediction (on-policy), Monte Carlo control with exploring starts, and
# off-policy estimation via ordinary vs weighted importance sampling.
#
# Input: none. Output: value-surface and optimal-policy figures.


#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Nicky van Foreest(vanforeest@gmail.com)                        #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# actions: hit or stand
ACTION_HIT = 0
ACTION_STAND = 1  #  "strike" in the book
ACTIONS = [ACTION_HIT, ACTION_STAND]

# policy for player
POLICY_PLAYER = np.zeros(22, dtype=np.int)
for i in range(12, 20):
    POLICY_PLAYER[i] = ACTION_HIT
POLICY_PLAYER[20] = ACTION_STAND
POLICY_PLAYER[21] = ACTION_STAND

# function form of target policy of player
def target_policy_player(usable_ace_player, player_sum, dealer_card):
    # TODO: implement target_policy_player
    raise NotImplementedError

# function form of behavior policy of player
def behavior_policy_player(usable_ace_player, player_sum, dealer_card):
    # TODO: implement behavior_policy_player
    raise NotImplementedError

# policy for dealer
POLICY_DEALER = np.zeros(22)
for i in range(12, 17):
    POLICY_DEALER[i] = ACTION_HIT
for i in range(17, 22):
    POLICY_DEALER[i] = ACTION_STAND

# get a new card
def get_card():
    # TODO: implement get_card
    raise NotImplementedError

# get the value of a card (11 for ace).
def card_value(card_id):
    # TODO: implement card_value
    raise NotImplementedError

# play a game
# @policy_player: specify policy for player
# @initial_state: [whether player has a usable Ace, sum of player's cards, one card of dealer]
# @initial_action: the initial action
def play(policy_player, initial_state=None, initial_action=None):
    # player status

    # sum of player
    # TODO: implement play
    raise NotImplementedError

# Monte Carlo Sample with On-Policy
def monte_carlo_on_policy(episodes):
    # TODO: implement monte_carlo_on_policy
    raise NotImplementedError

# Monte Carlo with Exploring Starts
def monte_carlo_es(episodes):
    # (playerSum, dealerCard, usableAce, action)
    # TODO: implement monte_carlo_es
    raise NotImplementedError

# Monte Carlo Sample with Off-Policy
def monte_carlo_off_policy(episodes):
    # TODO: implement monte_carlo_off_policy
    raise NotImplementedError

def figure_5_1():
    # TODO: implement figure_5_1
    raise NotImplementedError

def figure_5_2():
    # TODO: implement figure_5_2
    raise NotImplementedError

def figure_5_3():
    # TODO: implement figure_5_3
    raise NotImplementedError


if __name__ == '__main__':
    figure_5_1()
    figure_5_2()
    figure_5_3()
