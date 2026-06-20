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


# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # TODO: implement GPTDatasetV1.__init__
        raise NotImplementedError

    def __len__(self):
        # TODO: implement GPTDatasetV1.__len__
        raise NotImplementedError

    def __getitem__(self, idx):
        # TODO: implement GPTDatasetV1.__getitem__
        raise NotImplementedError


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):

    # Initialize the tokenizer
    # TODO: implement create_dataloader_v1
    raise NotImplementedError
