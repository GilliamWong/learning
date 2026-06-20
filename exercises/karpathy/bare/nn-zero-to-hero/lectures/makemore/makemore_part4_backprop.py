# Lecture: makemore, part 4 -- becoming a backprop ninja.
# Manually backpropagate through the entire MLP + BatchNorm model with no
# autograd, deriving the gradient of every intermediate tensor by hand.

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
