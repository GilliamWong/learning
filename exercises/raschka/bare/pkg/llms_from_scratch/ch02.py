# ch02.py -- turning raw text into batched training data for an LLM.
#
# Implement GPTDatasetV1 (given a long text and a tokenizer, yield overlapping
# (input, target) chunks via a sliding window of a chosen max length and stride,
# where the target is the input shifted by one token) and create_dataloader_v1
# (wrap that dataset in a torch DataLoader).
#
# Input:    a raw text string, a (BPE) tokenizer, windowing and batch settings.
# Output:   batches of (input_ids, target_ids) integer tensors, each
#           (batch, max_length).
# Behavior: tokenize once, then index sliding windows for next-token prediction.

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
