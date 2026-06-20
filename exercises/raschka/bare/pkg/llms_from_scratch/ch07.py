# ch07.py -- instruction finetuning (Alpaca-style).
#
# Implement format_input (render an instruction + optional input into the prompt
# style the model is trained to follow), InstructionDataset (pre-tokenize
# formatted prompt+response examples), and custom_collate_fn (pad a batch to
# equal length and build target ids shifted by one, with padding -- and
# optionally the prompt -- masked out of the loss).
#
# Input:    a list of instruction/response dicts and a tokenizer.
# Output:   batched (input_ids, target_ids) tensors ready for finetuning.
# Behavior: targets are inputs shifted by one; an ignore index removes padded
#           (and optionally prompt) positions from the loss.

import json
import os
import psutil
import requests
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
