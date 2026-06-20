# Lecture: makemore, part 2 -- a multi-layer perceptron language model.
# Embed a context of characters and predict the next one with an MLP. Learn the
# training essentials: minibatches, learning-rate search, train/dev/test splits.


# ======================================================================
# Auto-converted from notebook: makemore_part2_mlp.ipynb
# EXERCISE SKELETON — code cells stripped to TODOs.
# Markdown narration and imports are kept as guidance. Fill in the rest.
# '# %%' markers let you run this cell-by-cell in VS Code / Jupyter.
# ======================================================================

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
  #print(w)
# TODO: implement this cell (see solution / lecture video)
    #print(''.join(itos[i] for i in context), '--->', itos[ix])
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# build the dataset
# TODO: implement this cell (see solution / lecture video)
    #print(w)
# TODO: implement this cell (see solution / lecture video)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
# TODO: implement this cell (see solution / lecture video)
import random
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
# ------------ now made respectable :) ---------------

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
  # minibatch construct
# TODO: implement this cell (see solution / lecture video)
  # forward pass
# TODO: implement this cell (see solution / lecture video)
  #print(loss.item())
  # backward pass
# TODO: implement this cell (see solution / lecture video)
  # update
  #lr = lrs[i]
# TODO: implement this cell (see solution / lecture video)
  # track stats
  #lri.append(lre[i])
# TODO: implement this cell (see solution / lecture video)
#print(loss.item())

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# visualize dimensions 0 and 1 of the embedding matrix C for all characters
# TODO: implement this cell (see solution / lecture video)

# %%
# training split, dev/validation split, test split
# 80%, 10%, 10%

# %%
# TODO: implement this cell (see solution / lecture video)

# %%
# sample from the model
# TODO: implement this cell (see solution / lecture video)
