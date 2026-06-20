# gamblers_problem.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The gambler's problem (Chapter 4, Figure 4.3). Value iteration on a
# coin-flip betting MDP.
#
# Input: none. Output: value estimates across sweeps and the final
# capital->stake policy.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
