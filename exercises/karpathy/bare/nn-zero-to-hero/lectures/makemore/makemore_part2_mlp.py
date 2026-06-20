# Lecture: makemore, part 2 -- a multi-layer perceptron language model.
# Embed a context of characters and predict the next one with an MLP. Learn the
# training essentials: minibatches, learning-rate search, train/dev/test splits.

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
