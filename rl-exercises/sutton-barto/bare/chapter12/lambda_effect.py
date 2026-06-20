# lambda_effect.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The effect of the trace-decay parameter lambda (Chapter 12). Sweep lambda
# across the chapter's tasks/algorithms to show its influence on performance.
#
# Input: none. Output: performance-vs-lambda curves.

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor, log
from itertools import zip_longest
