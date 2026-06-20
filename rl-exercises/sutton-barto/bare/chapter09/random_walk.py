# random_walk.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The 1000-state random walk (Chapter 9, Figures 9.1/9.2/9.5/9.10). On-policy
# prediction with function approximation: gradient Monte Carlo and
# semi-gradient n-step TD using state aggregation, polynomial/Fourier bases,
# and tile coding.
#
# Input: none. Output: approximate value functions and error curves.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
