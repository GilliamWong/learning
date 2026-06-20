# Lecture: micrograd, part 1 -- backpropagation from scratch.
# Build a scalar autograd `Value` (its data, its grad, and the local backward of
# each operation) and see how reverse-mode differentiation walks the expression
# graph. Follow the lecture video and rebuild each cell yourself.

import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
