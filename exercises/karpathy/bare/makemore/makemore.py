# makemore.py -- character-level language models over a list of words.
#
# Read a file of words (one per line, e.g. names.txt) and train a model to
# generate new, similar words one character at a time.
#
# Implement: the dataset (words <-> integer index sequences, with start/end
#   tokens), the model zoo -- Bigram, MLP, RNN/GRU, Transformer (and BoW
#   baselines), each mapping character-index sequences to next-character logits
#   and an optional loss -- and autoregressive sampling of new sequences.
# Input:    a words file; CLI flags select the model and hyperparameters.
# Output:   a trained checkpoint and freshly sampled words.
# Behavior: batch -> forward to logits + loss -> backprop -> optimizer step,
#           with periodic evaluation and sampling.

import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
