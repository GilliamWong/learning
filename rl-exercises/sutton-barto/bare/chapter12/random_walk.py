# random_walk.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The 19-state random walk with eligibility traces (Chapter 12, Figures
# 12.3/12.6/12.8). Implement the offline lambda-return algorithm, TD(lambda),
# and true online TD(lambda).
#
# Input: none. Output: RMS-error vs (alpha, lambda) curves.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
