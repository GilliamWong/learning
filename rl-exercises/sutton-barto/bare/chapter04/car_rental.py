# car_rental.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Jack's car rental (Chapter 4, Figure 4.2). Policy iteration on a
# two-location rental MDP with Poisson demand and returns.
#
# Input: none. Output: the sequence of improved policies and the final value
# function.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson
