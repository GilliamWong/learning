# random_walk.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The 5-state random walk (Chapter 6, Example 6.2 / Figure 6.2). Compare TD(0)
# and constant-alpha Monte Carlo value estimation, including batch updating.
#
# Input: none. Output: learned values and RMS-error curves.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
