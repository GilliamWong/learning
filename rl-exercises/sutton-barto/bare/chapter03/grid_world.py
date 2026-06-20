# grid_world.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The 5x5 gridworld (Chapter 3, Figures 3.2 & 3.5). Compute the state-value
# function under the equiprobable random policy and the optimal value
# function/policy by solving the Bellman expectation and optimality equations.
#
# Input: none. Output: value-grid figures.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table
