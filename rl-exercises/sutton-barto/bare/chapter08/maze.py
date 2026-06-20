# maze.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The Dyna maze (Chapter 8, Figures 8.2/8.4/8.5, Example 8.4). Model-based RL:
# Dyna-Q and Dyna-Q+ with planning, plus prioritized sweeping, on gridworld
# mazes (including blocking/shortcut changes).
#
# Input: none. Output: learning curves vs number of planning steps.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq
from copy import deepcopy
