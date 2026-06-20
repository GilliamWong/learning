# access_control.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The access-control queuing task (Chapter 10, Figure 10.5). Average-reward
# control with differential semi-gradient Sarsa.
#
# Input: none. Output: the learned value/policy over free servers and
# priorities.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from math import floor
import seaborn as sns
