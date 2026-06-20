# Lecture: makemore, part 5 -- a WaveNet-style hierarchical conv net.
# Grow the MLP into a deeper, tree-like model that fuses characters in stages,
# and practice using torch.nn containers cleanly.

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
