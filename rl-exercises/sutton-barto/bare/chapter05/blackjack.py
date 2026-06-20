# blackjack.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Blackjack via Monte Carlo (Chapter 5, Figures 5.1-5.3). Implement Monte
# Carlo prediction (on-policy), Monte Carlo control with exploring starts, and
# off-policy estimation via ordinary vs weighted importance sampling.
#
# Input: none. Output: value-surface and optimal-policy figures.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
