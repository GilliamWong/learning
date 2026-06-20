# model.py -- a minimal GPT (decoder-only Transformer) language model in PyTorch.
#
# Implement causal (masked) multi-head self-attention, a Transformer block
# (attention + MLP with residual connections and layer norm), and the full GPT:
# token + positional embeddings, a stack of blocks, a final norm, and a linear
# head to vocabulary logits -- plus pretrained-GPT-2 loading, optimizer setup,
# and autoregressive generation.
#
# Input:    a config object and an integer token tensor of shape (batch, seq).
# Output:   logits of shape (batch, seq, vocab_size); with targets, also a
#           cross-entropy loss; generate() returns the extended token sequence.
# Behavior: a forward pass embeds tokens, applies the blocks causally, and
#           projects to vocabulary logits.

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from mingpt.utils import CfgNode as CN
