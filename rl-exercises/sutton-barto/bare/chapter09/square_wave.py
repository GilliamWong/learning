# square_wave.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The square-wave coarse-coding example (Chapter 9, Figure 9.8). Show how
# feature width affects generalization and asymptotic accuracy when
# approximating a square wave.
#
# Input: none. Output: learned approximations for several feature widths.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
