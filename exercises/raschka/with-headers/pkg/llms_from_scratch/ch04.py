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


# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from .ch03 import MultiHeadAttention, PyTorchMultiHeadAttention
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        # TODO: implement LayerNorm.__init__
        raise NotImplementedError

    def forward(self, x):
        # TODO: implement LayerNorm.forward
        raise NotImplementedError


class GELU(nn.Module):
    def __init__(self):
        # TODO: implement GELU.__init__
        raise NotImplementedError

    def forward(self, x):
        # TODO: implement GELU.forward
        raise NotImplementedError


class FeedForward(nn.Module):
    def __init__(self, cfg):
        # TODO: implement FeedForward.__init__
        raise NotImplementedError

    def forward(self, x):
        # TODO: implement FeedForward.forward
        raise NotImplementedError


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        # TODO: implement TransformerBlock.__init__
        raise NotImplementedError

    def forward(self, x):
        # Shortcut connection for attention block
        # TODO: implement TransformerBlock.forward
        raise NotImplementedError


class GPTModel(nn.Module):
    def __init__(self, cfg):
        # TODO: implement GPTModel.__init__
        raise NotImplementedError

    def forward(self, in_idx):
        # TODO: implement GPTModel.forward
        raise NotImplementedError


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    # TODO: implement generate_text_simple
    raise NotImplementedError

######################
# Bonus
######################


class FeedForwardFast(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlockFast(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = PyTorchMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForwardFast(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModelFast(nn.Module):
    """
    A faster variant of GPTModel optimized for training speed.

    This version is only marginally faster on CPU (~1.02x) but significantly
    faster on GPU (~2.05x) during training, thanks to optimized CUDA kernels
    and FlashAttention support.

    Key differences from the original GPTModel:
    1. Uses PyTorch's built-in LayerNorm instead of a custom implementation.
    2. Uses PyTorch's built-in GELU instead of a custom implementation.
    3. Uses PyTorch's scaled_dot_product_attention instead of a custom MultiHeadAttention.
    4. Automatically enables FlashAttention on compatible GPUs.
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlockFast(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
