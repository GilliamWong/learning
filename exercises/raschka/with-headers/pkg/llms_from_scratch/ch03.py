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


# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import torch
import torch.nn as nn


class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        # TODO: implement SelfAttention_v1.__init__
        raise NotImplementedError

    def forward(self, x):
        # TODO: implement SelfAttention_v1.forward
        raise NotImplementedError


class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        # TODO: implement SelfAttention_v2.__init__
        raise NotImplementedError

    def forward(self, x):
        # TODO: implement SelfAttention_v2.forward
        raise NotImplementedError


class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        # TODO: implement CausalAttention.__init__
        raise NotImplementedError

    def forward(self, x):
        # TODO: implement CausalAttention.forward
        raise NotImplementedError


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        # TODO: implement MultiHeadAttentionWrapper.__init__
        raise NotImplementedError

    def forward(self, x):
        # TODO: implement MultiHeadAttentionWrapper.forward
        raise NotImplementedError


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        # TODO: implement MultiHeadAttention.__init__
        raise NotImplementedError

    def forward(self, x):
        # TODO: implement MultiHeadAttention.forward
        raise NotImplementedError


######################
# Bonus
######################


class PyTorchMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec
