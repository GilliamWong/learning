# grid_world.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The 4x4 gridworld (Chapter 4, Figure 4.1). Iterative policy evaluation of
# the equiprobable random policy until convergence.
#
# Input: none. Output: the value function across sweeps.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table
