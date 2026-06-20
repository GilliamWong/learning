# mountain_car.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Mountain Car with eligibility traces (Chapter 12, Figures 12.10 & 12.11).
# Sarsa(lambda) with replacing traces and tile coding.
#
# Input: none. Output: learning curves comparing trace variants.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import floor
from tqdm import tqdm
