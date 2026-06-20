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


"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        # TODO: implement NewGELU.forward
        raise NotImplementedError

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        # TODO: implement CausalSelfAttention.__init__
        raise NotImplementedError

    def forward(self, x):
        # TODO: implement CausalSelfAttention.forward
        raise NotImplementedError

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        # TODO: implement Block.__init__
        raise NotImplementedError

    def forward(self, x):
        # TODO: implement Block.forward
        raise NotImplementedError

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        # TODO: implement GPT.get_default_config
        raise NotImplementedError

    def __init__(self, config):
        # TODO: implement GPT.__init__
        raise NotImplementedError

    def _init_weights(self, module):
        # TODO: implement GPT._init_weights
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        # TODO: implement GPT.from_pretrained
        raise NotImplementedError

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        # TODO: implement GPT.configure_optimizers
        raise NotImplementedError

    def forward(self, idx, targets=None):
        # TODO: implement GPT.forward
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # TODO: implement GPT.generate
        raise NotImplementedError
