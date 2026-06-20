# Lecture: micrograd, part 2 -- a neural net on top of Value.
# Use Value to build neurons / layers / an MLP, define a loss, and train by hand:
# forward, backward, nudge the parameters along their gradients, repeat.

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import torch
