# ch05.py -- pretraining the GPT and loading pretrained weights.
#
# Implement the training / generation utilities: per-batch and per-loader
# cross-entropy loss, the training loop with periodic evaluation, autoregressive
# generation with temperature and top-k, tokenizer <-> tensor helpers, and
# copying OpenAI GPT-2 parameters into your GPTModel.
#
# Input:    a GPTModel, data loaders, optimizer/device, a tokenizer, and (for
#           weight loading) a dict of pretrained GPT-2 parameters.
# Output:   training/validation loss histories; generated token tensors / text.
# Behavior: standard next-token-prediction training; sampling decodes one token
#           at a time from the model's logits.

from .ch04 import generate_text_simple
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import requests
import torch
from tqdm import tqdm
