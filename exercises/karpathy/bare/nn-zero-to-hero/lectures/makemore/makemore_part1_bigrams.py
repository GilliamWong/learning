# Lecture: makemore, part 1 -- the bigram character language model.
# Count character bigrams from names.txt, turn the counts into probabilities and
# sample new names, then re-derive the same model as a 1-layer neural net trained
# with negative log-likelihood.

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
