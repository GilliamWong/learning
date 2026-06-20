# square_wave.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The square-wave coarse-coding example (Chapter 9, Figure 9.8). Show how
# feature width affects generalization and asymptotic accuracy when
# approximating a square wave.
#
# Input: none. Output: learned approximations for several feature widths.


#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# wrapper class for an interval
# readability is more important than efficiency, so I won't use many tricks
class Interval:
    # [@left, @right)
    def __init__(self, left, right):
        # TODO: implement Interval.__init__
        raise NotImplementedError

    # whether a point is in this interval
    def contain(self, x):
        # TODO: implement Interval.contain
        raise NotImplementedError

    # length of this interval
    def size(self):
        # TODO: implement Interval.size
        raise NotImplementedError

# domain of the square wave, [0, 2)
DOMAIN = Interval(0.0, 2.0)

# square wave function
def square_wave(x):
    # TODO: implement square_wave
    raise NotImplementedError

# get @n samples randomly from the square wave
def sample(n):
    # TODO: implement sample
    raise NotImplementedError

# wrapper class for value function
class ValueFunction:
    # @domain: domain of this function, an instance of Interval
    # @alpha: basic step size for one update
    def __init__(self, feature_width, domain=DOMAIN, alpha=0.2, num_of_features=50):
        # TODO: implement ValueFunction.__init__
        raise NotImplementedError

    # for point @x, return the indices of corresponding feature windows
    def get_active_features(self, x):
        # TODO: implement ValueFunction.get_active_features
        raise NotImplementedError

    # estimate the value for point @x
    def value(self, x):
        # TODO: implement ValueFunction.value
        raise NotImplementedError

    # update weights given sample of point @x
    # @delta: y - x
    def update(self, delta, x):
        # TODO: implement ValueFunction.update
        raise NotImplementedError

# train @value_function with a set of samples @samples
def approximate(samples, value_function):
    # TODO: implement approximate
    raise NotImplementedError

# Figure 9.8
def figure_9_8():
    # TODO: implement figure_9_8
    raise NotImplementedError

if __name__ == '__main__':
    figure_9_8()