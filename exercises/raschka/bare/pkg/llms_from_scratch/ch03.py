# ch03.py -- self-attention and multi-head attention.
#
# Implement the attention mechanisms the GPT is built from: simplified
# self-attention (v1 with raw parameters, v2 with nn.Linear), causal attention
# (future positions masked, with dropout), and multi-head attention (several
# causal heads in parallel, then projected).
#
# Input:    token embeddings of shape (batch, num_tokens, d_in).
# Output:   a context tensor of shape (batch, num_tokens, d_out): each position
#           is a weighted blend of the current and earlier positions.
# Behavior: scaled dot-product attention with a causal mask; multi-head splits
#           the projection into independent subspaces and concatenates them.

import torch
import torch.nn as nn
