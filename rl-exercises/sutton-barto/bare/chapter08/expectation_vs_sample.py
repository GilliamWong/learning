# expectation_vs_sample.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Expected vs sample updates (Chapter 8, Figure 8.7). Compare the efficiency
# of expected and sample updates as a function of branching factor.
#
# Input: none. Output: error-vs-computation curves.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
