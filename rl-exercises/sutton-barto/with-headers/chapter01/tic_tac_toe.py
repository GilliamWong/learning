# tic_tac_toe.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Tic-Tac-Toe (Chapter 1, the introductory example). Two agents learn to play
# by repeated self-play, updating a state-value table toward game outcomes
# (temporal-difference learning); a trained agent can then play a human.
#
# Input: none (self-play episodes). Output: a learned value table + win/draw
# statistics (and an interactive game).


#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS


class State:
    def __init__(self):
        # the board is represented by an n * n array,
        # 1 represents a chessman of the player who moves first,
        # -1 represents a chessman of another player
        # 0 represents an empty position
        # TODO: implement State.__init__
        raise NotImplementedError

    # compute the hash value for one state, it's unique
    def hash(self):
        # TODO: implement State.hash
        raise NotImplementedError

    # check whether a player has won the game, or it's a tie
    def is_end(self):
        # TODO: implement State.is_end
        raise NotImplementedError

    # @symbol: 1 or -1
    # put chessman symbol in position (i, j)
    def next_state(self, i, j, symbol):
        # TODO: implement State.next_state
        raise NotImplementedError

    # print the board
    def print_state(self):
        # TODO: implement State.print_state
        raise NotImplementedError


def get_all_states_impl(current_state, current_symbol, all_states):
    # TODO: implement get_all_states_impl
    raise NotImplementedError


def get_all_states():
    # TODO: implement get_all_states
    raise NotImplementedError


# all possible board configurations
all_states = get_all_states()


class Judger:
    # @player1: the player who will move first, its chessman will be 1
    # @player2: another player with a chessman -1
    def __init__(self, player1, player2):
        # TODO: implement Judger.__init__
        raise NotImplementedError

    def reset(self):
        # TODO: implement Judger.reset
        raise NotImplementedError

    def alternate(self):
        # TODO: implement Judger.alternate
        raise NotImplementedError

    # @print_state: if True, print each board during the game
    def play(self, print_state=False):
        # TODO: implement Judger.play
        raise NotImplementedError


# AI player
class Player:
    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    def __init__(self, step_size=0.1, epsilon=0.1):
        # TODO: implement Player.__init__
        raise NotImplementedError

    def reset(self):
        # TODO: implement Player.reset
        raise NotImplementedError

    def set_state(self, state):
        # TODO: implement Player.set_state
        raise NotImplementedError

    def set_symbol(self, symbol):
        # TODO: implement Player.set_symbol
        raise NotImplementedError

    # update value estimation
    def backup(self):
        # TODO: implement Player.backup
        raise NotImplementedError

    # choose an action based on the state
    def act(self):
        # TODO: implement Player.act
        raise NotImplementedError

    def save_policy(self):
        # TODO: implement Player.save_policy
        raise NotImplementedError

    def load_policy(self):
        # TODO: implement Player.load_policy
        raise NotImplementedError


# human interface
# input a number to put a chessman
# | q | w | e |
# | a | s | d |
# | z | x | c |
class HumanPlayer:
    def __init__(self, **kwargs):
        # TODO: implement HumanPlayer.__init__
        raise NotImplementedError

    def reset(self):
        # TODO: implement HumanPlayer.reset
        raise NotImplementedError

    def set_state(self, state):
        # TODO: implement HumanPlayer.set_state
        raise NotImplementedError

    def set_symbol(self, symbol):
        # TODO: implement HumanPlayer.set_symbol
        raise NotImplementedError

    def act(self):
        # TODO: implement HumanPlayer.act
        raise NotImplementedError


def train(epochs, print_every_n=500):
    # TODO: implement train
    raise NotImplementedError


def compete(turns):
    # TODO: implement compete
    raise NotImplementedError


# The game is a zero sum game. If both players are playing with an optimal strategy, every game will end in a tie.
# So we test whether the AI can guarantee at least a tie if it goes second.
def play():
    # TODO: implement play
    raise NotImplementedError


if __name__ == '__main__':
    train(int(1e5))
    compete(int(1e3))
    play()
