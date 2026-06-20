# short_corridor.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The short-corridor gridworld (Chapter 13, Figures 13.1 & 13.2).
# Policy-gradient control with REINFORCE (and REINFORCE with baseline) on a
# small problem whose optimal policy is stochastic.
#
# Input: none. Output: total-reward-vs-episode curves.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
