# ch04.py -- assembling the full GPT model.
#
# Building on the chapter-3 attention, implement LayerNorm, the GELU activation,
# a position-wise FeedForward, a TransformerBlock (norm -> attention -> residual,
# norm -> feed-forward -> residual), the GPTModel (token + positional embeddings,
# a stack of blocks, final norm, linear head to logits), and greedy
# generate_text_simple.
#
# Input:    a config dict (vocab_size, context_length, emb_dim, n_heads,
#           n_layers, drop_rate, qkv_bias) and an integer token tensor
#           (batch, num_tokens).
# Output:   logits of shape (batch, num_tokens, vocab_size); generate returns the
#           input extended by max_new_tokens ids.
# Behavior: a forward pass embeds, runs the blocks, and projects to logits.

from .ch03 import MultiHeadAttention, PyTorchMultiHeadAttention
import torch
import torch.nn as nn
