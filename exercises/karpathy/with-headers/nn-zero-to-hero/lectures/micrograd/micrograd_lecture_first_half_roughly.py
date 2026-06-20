# Lecture: micrograd, part 1 -- backpropagation from scratch.
# Build a scalar autograd `Value` (its data, its grad, and the local backward of
# each operation) and see how reverse-mode differentiation walks the expression
# graph. Follow the lecture video and rebuild each cell yourself.


# ======================================================================
# Auto-converted from notebook: micrograd_lecture_first_half_roughly.ipynb
# EXERCISE SKELETON — code cells stripped to TODOs.
# Markdown narration and imports are kept as guidance. Fill in the rest.
# '# %%' markers let you run this cell-by-cell in VS Code / Jupyter.
# ======================================================================

# %%
import math
import numpy as np
import matplotlib.pyplot as plt
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# les get more complex
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)
# inputs
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
from graphviz import Digraph
# TODO: implement this cell (see solution / lecture video)
  # builds a set of all nodes and edges in a graph
# TODO: implement this cell (see solution / lecture video)
    # for any value in the graph, create a rectangular ('record') node for it
# TODO: implement this cell (see solution / lecture video)
      # if this value is a result of some operation, create an op node for it
# TODO: implement this cell (see solution / lecture video)
      # and connect this node to it
# TODO: implement this cell (see solution / lecture video)
    # connect n1 to the op node of n2
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# inputs x1,x2
# TODO: implement this cell (see solution / lecture video)
# weights w1,w2
# TODO: implement this cell (see solution / lecture video)
# bias of the neuron
# TODO: implement this cell (see solution / lecture video)
# x1*w1 + x2*w2 + b
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# o = tanh(n)
# do/dn = 1 - o**2

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)
