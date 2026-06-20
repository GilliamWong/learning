# Lecture: makemore, part 4 -- becoming a backprop ninja.
# Manually backpropagate through the entire MLP + BatchNorm model with no
# autograd, deriving the gradient of every intermediate tensor by hand.


# ======================================================================
# Auto-converted from notebook: makemore_part4_backprop.ipynb
# EXERCISE SKELETON — code cells stripped to TODOs.
# Markdown narration and imports are kept as guidance. Fill in the rest.
# '# %%' markers let you run this cell-by-cell in VS Code / Jupyter.
# ======================================================================

# %% [markdown]
# ## makemore: becoming a backprop ninja
#
# swole doge style

# %%
# there no change change in the first several cells from last lecture

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
# build the dataset
# TODO: implement this cell (see solution / lecture video)
import random
# TODO: implement this cell (see solution / lecture video)

# %%
# ok biolerplate done, now we get to the action:

# %%
# utility function we will use later when comparing manual gradients to PyTorch gradients
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)
# Layer 1
# TODO: implement this cell (see solution / lecture video)
# Layer 2
# TODO: implement this cell (see solution / lecture video)
# BatchNorm parameters
# TODO: implement this cell (see solution / lecture video)
# Note: I am initializating many of these parameters in non-standard ways
# because sometimes initializating with e.g. all zeros could mask an incorrect
# implementation of the backward pass.
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)
# construct a minibatch
# TODO: implement this cell (see solution / lecture video)

# %%
# forward pass, "chunkated" into smaller steps that are possible to backward one at a time
# TODO: implement this cell (see solution / lecture video)
# Linear layer 1
# TODO: implement this cell (see solution / lecture video)
# BatchNorm layer
# TODO: implement this cell (see solution / lecture video)
# Non-linearity
# TODO: implement this cell (see solution / lecture video)
# Linear layer 2
# TODO: implement this cell (see solution / lecture video)
# cross entropy loss (same as F.cross_entropy(logits, Yb))
# TODO: implement this cell (see solution / lecture video)
# PyTorch backward pass
# TODO: implement this cell (see solution / lecture video)

# %%
# Exercise 1: backprop through the whole thing manually, 
# backpropagating through exactly all of the variables 
# as they are defined in the forward pass above, one by one
# TODO: implement this cell (see solution / lecture video)

# %%
# Exercise 2: backprop through cross_entropy but all in one go
# to complete this challenge look at the mathematical expression of the loss,
# take the derivative, simplify the expression, and just write it out
# forward pass
# before:
# logit_maxes = logits.max(1, keepdim=True).values
# norm_logits = logits - logit_maxes # subtract max for numerical stability
# counts = norm_logits.exp()
# counts_sum = counts.sum(1, keepdims=True)
# counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
# probs = counts * counts_sum_inv
# logprobs = probs.log()
# loss = -logprobs[range(n), Yb].mean()
# now:
# TODO: implement this cell (see solution / lecture video)

# %%
# backward pass
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
# Exercise 3: backprop through batchnorm but all in one go
# to complete this challenge look at the mathematical expression of the output of batchnorm,
# take the derivative w.r.t. its input, simplify the expression, and just write it out
# forward pass
# before:
# bnmeani = 1/n*hprebn.sum(0, keepdim=True)
# bndiff = hprebn - bnmeani
# bndiff2 = bndiff**2
# bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
# bnvar_inv = (bnvar + 1e-5)**-0.5
# bnraw = bndiff * bnvar_inv
# hpreact = bngain * bnraw + bnbias
# now:
# TODO: implement this cell (see solution / lecture video)

# %%
# backward pass
# before we had:
# dbnraw = bngain * dhpreact
# dbndiff = bnvar_inv * dbnraw
# dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
# dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv
# dbndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2) * dbnvar
# dbndiff += (2*bndiff) * dbndiff2
# dhprebn = dbndiff.clone()
# dbnmeani = (-dbndiff).sum(0)
# dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)
# calculate dhprebn given dhpreact (i.e. backprop through the batchnorm)
# (you'll also need to use some of the variables from the forward pass up above)
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# Exercise 4: putting it all together!
# Train the MLP neural net with your own backward pass
# init
# TODO: implement this cell (see solution / lecture video)
# Layer 1
# TODO: implement this cell (see solution / lecture video)
# Layer 2
# TODO: implement this cell (see solution / lecture video)
# BatchNorm parameters
# TODO: implement this cell (see solution / lecture video)
# same optimization as last time
# TODO: implement this cell (see solution / lecture video)
# use this context manager for efficiency once your backward pass is written (TODO)
# TODO: implement this cell (see solution / lecture video)
  # kick off optimization
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
    #loss.backward() # use this for correctness comparisons, delete it later!
    # manual backprop! #swole_doge_meme
    # -----------------
# TODO: implement this cell (see solution / lecture video)
    # 2nd layer backprop
# TODO: implement this cell (see solution / lecture video)
    # tanh
# TODO: implement this cell (see solution / lecture video)
    # batchnorm backprop
# TODO: implement this cell (see solution / lecture video)
    # 1st layer
# TODO: implement this cell (see solution / lecture video)
    # embedding
# TODO: implement this cell (see solution / lecture video)
    # -----------------
    # update
# TODO: implement this cell (see solution / lecture video)
      #p.data += -lr * p.grad # old way of cheems doge (using PyTorch grad from .backward())
# TODO: implement this cell (see solution / lecture video)
    # track stats
# TODO: implement this cell (see solution / lecture video)
  #   if i >= 100: # TODO: delete early breaking when you're ready to train the full net
  #     break

# %%
# useful for checking your gradients
# for p,g in zip(parameters, grads):
#   cmp(str(tuple(p.shape)), g, p)

# %%
# calibrate the batch norm at the end of training
# TODO: implement this cell (see solution / lecture video)
  # pass the training set through
# TODO: implement this cell (see solution / lecture video)
  # measure the mean/std over the entire training set
# TODO: implement this cell (see solution / lecture video)

# %%
# evaluate train and val loss
# TODO: implement this cell (see solution / lecture video)

# %%
# I achieved:
# train 2.0718822479248047
# val 2.1162495613098145

# %%
# sample from the model
# TODO: implement this cell (see solution / lecture video)
      # ------------
      # forward pass:
      # Embedding
# TODO: implement this cell (see solution / lecture video)
      # ------------
      # Sample
# TODO: implement this cell (see solution / lecture video)
