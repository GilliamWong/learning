# Lecture: makemore, part 3 -- activations, gradients, and BatchNorm.
# Diagnose a deeper network by inspecting the statistics of its activations and
# gradients, fix the initialization, and add BatchNorm to keep training stable.

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import scipy.stats as stats
import numpy as np
