# maximization_bias.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Maximization bias (Chapter 6, Figure 6.7). Compare Q-learning and Double
# Q-learning on the small MDP that exposes maximization bias.
#
# Input: none. Output: %-left-action vs episode curves.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
