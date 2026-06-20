# tic_tac_toe.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Tic-Tac-Toe (Chapter 1, the introductory example). Two agents learn to play
# by repeated self-play, updating a state-value table toward game outcomes
# (temporal-difference learning); a trained agent can then play a human.
#
# Input: none (self-play episodes). Output: a learned value table + win/draw
# statistics (and an interactive game).

import numpy as np
import pickle
