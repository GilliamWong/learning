# trajectory_sampling.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Trajectory sampling (Chapter 8, Figure 8.8). Compare on-policy trajectory
# sampling against a uniform update distribution on random MDPs.
#
# Input: none. Output: value-of-start-state vs computation.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
