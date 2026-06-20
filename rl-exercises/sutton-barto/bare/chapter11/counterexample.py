# counterexample.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Baird's counterexample (Chapter 11, Figures 11.2/11.6/11.7). Demonstrate
# off-policy divergence of semi-gradient TD, and the stabler behavior of
# TDC/Gradient-TD and Emphatic-TD.
#
# Input: none. Output: weight-trajectory curves.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d.axes3d import Axes3D
