# Lecture: makemore, part 1 -- the bigram character language model.
# Count character bigrams from names.txt, turn the counts into probabilities and
# sample new names, then re-derive the same model as a 1-layer neural net trained
# with negative log-likelihood.


# ======================================================================
# Auto-converted from notebook: makemore_part1_bigrams.ipynb
# EXERCISE SKELETON — code cells stripped to TODOs.
# Markdown narration and imports are kept as guidance. Fill in the rest.
# '# %%' markers let you run this cell-by-cell in VS Code / Jupyter.
# ======================================================================

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
import torch

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
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
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# 27, 27
# 27,  1

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# 27, 27
#  1, 27

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood
# log(a*b*c) = log(a) + log(b) + log(c)

# %%
# TODO: implement this cell (see solution / lecture video)
#for w in ["andrejq"]:
# TODO: implement this cell (see solution / lecture video)
    #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
# TODO: implement this cell (see solution / lecture video)

# %%
# create the training set of bigrams (x,y)
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
import torch.nn.functional as F
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
# (5, 27) @ (27, 27) -> (5, 27)

# %%
# SUMMARY ------------------------------>>>>

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# randomly initialize 27 neurons' weights. each neuron receives 27 inputs
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)
# btw: the last 2 lines here are together called a 'softmax'

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)
  # i-th bigram:
# TODO: implement this cell (see solution / lecture video)

# %%
# --------- !!! OPTIMIZATION !!! yay --------------

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# randomly initialize 27 neurons' weights. each neuron receives 27 inputs
# TODO: implement this cell (see solution / lecture video)

# %%
# forward pass
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# backward pass
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# --------- !!! OPTIMIZATION !!! yay, but this time actually --------------

# %%
# create the dataset
# TODO: implement this cell (see solution / lecture video)
# initialize the 'network'
# TODO: implement this cell (see solution / lecture video)

# %%
# gradient descent
# TODO: implement this cell (see solution / lecture video)
  # forward pass
# TODO: implement this cell (see solution / lecture video)
  # backward pass
# TODO: implement this cell (see solution / lecture video)
  # update
# TODO: implement this cell (see solution / lecture video)

# %%
# finally, sample from the 'neural net' model
# TODO: implement this cell (see solution / lecture video)
    # ----------
    # BEFORE:
    #p = P[ix]
    # ----------
    # NOW:
# TODO: implement this cell (see solution / lecture video)
    # ----------
# TODO: implement this cell (see solution / lecture video)
