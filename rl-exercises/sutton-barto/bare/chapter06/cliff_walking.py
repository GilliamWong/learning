# cliff_walking.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The cliff-walking task (Chapter 6, Figures 6.4 & 6.6). Compare Sarsa,
# Expected Sarsa, and Q-learning on a gridworld with a cliff edge.
#
# Input: none. Output: reward-per-episode curves and the greedy policies.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
