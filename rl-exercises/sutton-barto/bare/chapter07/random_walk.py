# random_walk.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The 19-state random walk (Chapter 7, Figure 7.2). n-step TD prediction:
# study how performance varies with the number of steps n and the step size
# alpha.
#
# Input: none. Output: RMS-error vs alpha curves for several n.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
