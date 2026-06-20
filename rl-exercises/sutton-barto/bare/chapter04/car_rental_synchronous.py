# car_rental_synchronous.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# Jack's car rental (Chapter 4, Figure 4.2), synchronous-update variant. Same
# policy-iteration problem with synchronous value sweeps.
#
# Input: none. Output: improved policies and the final value function.

import numpy as np
import matplotlib.pyplot as plt
import math
import tqdm
import multiprocessing as mp
from functools import partial
import time
import itertools
