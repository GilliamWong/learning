# infinite_variance.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The infinite-variance example (Chapter 5, Figure 5.4). Show that ordinary
# importance-sampling estimates can have infinite variance on a one-state
# problem.
#
# Input: none. Output: estimate trajectories across many runs.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
