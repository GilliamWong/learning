# Lecture: makemore, part 5 -- a WaveNet-style hierarchical conv net.
# Grow the MLP into a deeper, tree-like model that fuses characters in stages,
# and practice using torch.nn containers cleanly.


# ======================================================================
# Auto-converted from notebook: makemore_part5_cnn1.ipynb
# EXERCISE SKELETON — code cells stripped to TODOs.
# Markdown narration and imports are kept as guidance. Fill in the rest.
# '# %%' markers let you run this cell-by-cell in VS Code / Jupyter.
# ======================================================================

# %% [markdown]
# ## makemore: part 5

# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
# TODO: implement this cell (see solution / lecture video)

# %%
# read in all the words
# TODO: implement this cell (see solution / lecture video)

# %%
# build the vocabulary of characters and mappings to/from integers
# TODO: implement this cell (see solution / lecture video)

# %%
# shuffle up the words
import random
# TODO: implement this cell (see solution / lecture video)

# %%
# build the dataset
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# Near copy paste of the layers we have developed in Part 3
# -----------------------------------------------------------------------------------------------
# TODO: implement this cell (see solution / lecture video)
# -----------------------------------------------------------------------------------------------
# TODO: implement this cell (see solution / lecture video)
    # parameters (trained with backprop)
# TODO: implement this cell (see solution / lecture video)
    # buffers (trained with a running 'momentum update')
# TODO: implement this cell (see solution / lecture video)
    # calculate the forward pass
# TODO: implement this cell (see solution / lecture video)
    # update the buffers
# TODO: implement this cell (see solution / lecture video)
# -----------------------------------------------------------------------------------------------
# TODO: implement this cell (see solution / lecture video)
# -----------------------------------------------------------------------------------------------
# TODO: implement this cell (see solution / lecture video)
# -----------------------------------------------------------------------------------------------
# TODO: implement this cell (see solution / lecture video)
# -----------------------------------------------------------------------------------------------
# TODO: implement this cell (see solution / lecture video)
    # get parameters of all layers and stretch them out into one list
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# original network
# n_embd = 10 # the dimensionality of the character embedding vectors
# n_hidden = 300 # the number of neurons in the hidden layer of the MLP
# model = Sequential([
#   Embedding(vocab_size, n_embd),
#   FlattenConsecutive(8), Linear(n_embd * 8, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#   Linear(n_hidden, vocab_size),
# ])
# hierarchical network
# TODO: implement this cell (see solution / lecture video)
# parameter init
# TODO: implement this cell (see solution / lecture video)

# %%
# same optimization as last time
# TODO: implement this cell (see solution / lecture video)
  # minibatch construct
# TODO: implement this cell (see solution / lecture video)
  # forward pass
# TODO: implement this cell (see solution / lecture video)
  # backward pass
# TODO: implement this cell (see solution / lecture video)
  # update: simple SGD
# TODO: implement this cell (see solution / lecture video)
  # track stats
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# put layers into eval mode (needed for batchnorm especially)
# TODO: implement this cell (see solution / lecture video)

# %%
# evaluate the loss
# TODO: implement this cell (see solution / lecture video)

# %% [markdown]
# ### performance log
#
# - original (3 character context + 200 hidden neurons, 12K params): train 2.058, val 2.105
# - context: 3 -> 8 (22K params): train 1.918, val 2.027
# - flat -> hierarchical (22K params): train 1.941, val 2.029
# - fix bug in batchnorm: train 1.912, val 2.022
# - scale up the network: n_embd 24, n_hidden 128 (76K params): train 1.769, val 1.993

# %%
# sample from the model
# TODO: implement this cell (see solution / lecture video)
      # forward pass the neural net
# TODO: implement this cell (see solution / lecture video)
      # sample from the distribution
# TODO: implement this cell (see solution / lecture video)
      # shift the context window and track the samples
# TODO: implement this cell (see solution / lecture video)
      # if we sample the special '.' token, break
# TODO: implement this cell (see solution / lecture video)

# %% [markdown]
# ### Next time:
# Why convolutions? Brief preview/hint

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# forward a single example:
# TODO: implement this cell (see solution / lecture video)

# %%
# forward all of them
# TODO: implement this cell (see solution / lecture video)

# %%
# convolution is a "for loop"
# allows us to forward Linear layers efficiently over space
