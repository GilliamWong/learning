# ten_armed_testbed.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The 10-armed bandit testbed (Chapter 2, Figures 2.1-2.6). Implement a
# k-armed bandit and the action-value methods that learn on it:
# epsilon-greedy, optimistic initial values, UCB, and gradient bandits.
#
# Input: none (parameters are module constants). Output: matplotlib figures of
# average reward and % optimal action.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
