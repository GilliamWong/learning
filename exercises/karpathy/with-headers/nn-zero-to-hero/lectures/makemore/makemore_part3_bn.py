# Lecture: makemore, part 3 -- activations, gradients, and BatchNorm.
# Diagnose a deeper network by inspecting the statistics of its activations and
# gradients, fix the initialization, and add BatchNorm to keep training stable.


# ======================================================================
# Auto-converted from notebook: makemore_part3_bn.ipynb
# EXERCISE SKELETON — code cells stripped to TODOs.
# Markdown narration and imports are kept as guidance. Fill in the rest.
# '# %%' markers let you run this cell-by-cell in VS Code / Jupyter.
# ======================================================================

# %% [markdown]
# # makemore: part 3

# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
# TODO: implement this cell (see solution / lecture video)

# %%
# read in all the words
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# build the vocabulary of characters and mappings to/from integers
# TODO: implement this cell (see solution / lecture video)

# %%
# build the dataset
# TODO: implement this cell (see solution / lecture video)
import random
# TODO: implement this cell (see solution / lecture video)

# %%
# MLP revisited
# TODO: implement this cell (see solution / lecture video)
#b1 = torch.randn(n_hidden,                        generator=g) * 0.01
# TODO: implement this cell (see solution / lecture video)
# BatchNorm parameters
# TODO: implement this cell (see solution / lecture video)

# %%
# same optimization as last time
# TODO: implement this cell (see solution / lecture video)
  # minibatch construct
# TODO: implement this cell (see solution / lecture video)
  # forward pass
# TODO: implement this cell (see solution / lecture video)
  # Linear layer
# TODO: implement this cell (see solution / lecture video)
  # BatchNorm layer
  # -------------------------------------------------------------
# TODO: implement this cell (see solution / lecture video)
  # -------------------------------------------------------------
  # Non-linearity
# TODO: implement this cell (see solution / lecture video)
  # backward pass
# TODO: implement this cell (see solution / lecture video)
  # update
# TODO: implement this cell (see solution / lecture video)
  # track stats
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# calibrate the batch norm at the end of training
# TODO: implement this cell (see solution / lecture video)
  # pass the training set through
# TODO: implement this cell (see solution / lecture video)
  # measure the mean/std over the entire training set
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)
  #hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias
# TODO: implement this cell (see solution / lecture video)

# %% [markdown]
# ## loss log
#
# ### original:
# train 2.1245384216308594
# val   2.168196439743042
#
# ### fix softmax confidently wrong:
# train 2.07
# val   2.13
#
# ### fix tanh layer too saturated at init:
# train 2.0355966091156006
# val   2.1026785373687744
#
# ### use semi-principled "kaiming init" instead of hacky init:
# train 2.0376641750335693
# val   2.106989622116089
#
# ### add batch norm layer
# train 2.0668270587921143
# val 2.104844808578491

# %%
# SUMMARY + PYTORCHIFYING -----------

# %%
# Let's train a deeper network
# The classes we create here are the same API as nn.Module in PyTorch
# TODO: implement this cell (see solution / lecture video)
    # parameters (trained with backprop)
# TODO: implement this cell (see solution / lecture video)
    # buffers (trained with a running 'momentum update')
# TODO: implement this cell (see solution / lecture video)
    # calculate the forward pass
# TODO: implement this cell (see solution / lecture video)
    # update the buffers
# TODO: implement this cell (see solution / lecture video)
# layers = [
#   Linear(n_embd * block_size, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, vocab_size),
# ]
# TODO: implement this cell (see solution / lecture video)
  # last layer: make less confident
# TODO: implement this cell (see solution / lecture video)
  #layers[-1].weight *= 0.1
  # all other layers: apply gain
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
  # update
# TODO: implement this cell (see solution / lecture video)
  # track stats
# TODO: implement this cell (see solution / lecture video)

# %%
# visualize histograms
# TODO: implement this cell (see solution / lecture video)

# %%
# visualize histograms
# TODO: implement this cell (see solution / lecture video)

# %%
# visualize histograms
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)
# put layers into eval mode
# TODO: implement this cell (see solution / lecture video)

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

# %%
# DONE; BONUS content below, not covered in video

# %%
# BatchNorm forward pass as a widget
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import scipy.stats as stats
import numpy as np
# TODO: implement this cell (see solution / lecture video)
  # plot 0
# TODO: implement this cell (see solution / lecture video)
  # plot the mean and std
# TODO: implement this cell (see solution / lecture video)
  # plot little lines connecting input and output
# TODO: implement this cell (see solution / lecture video)
  # plot the input and output values
# TODO: implement this cell (see solution / lecture video)
  # title
# TODO: implement this cell (see solution / lecture video)

# %%
# Linear: activation statistics of forward and backward pass
# TODO: implement this cell (see solution / lecture video)

# %%
# Linear + BatchNorm: activation statistics of forward and backward pass
# TODO: implement this cell (see solution / lecture video)
# linear layer ---
# TODO: implement this cell (see solution / lecture video)
# bn layer ---
# TODO: implement this cell (see solution / lecture video)
# ----
# TODO: implement this cell (see solution / lecture video)
