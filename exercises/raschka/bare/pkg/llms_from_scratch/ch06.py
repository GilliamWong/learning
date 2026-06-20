# ch06.py -- finetuning the GPT for text classification (spam vs. ham).
#
# Implement SpamDataset (tokenize and pad/truncate labelled messages to a fixed
# length, yielding (token_ids, label)), the classification loss/accuracy helpers
# and evaluation (read from the LAST token's logits), the finetuning loop, and a
# classify_review helper that runs the finetuned model on new text.
#
# Input:    a CSV of (text, label) rows, a tokenizer, and a GPT whose head has
#           been replaced for classification.
# Output:   a trained classifier; per-example spam / not-spam predictions; loss
#           and accuracy histories.
# Behavior: use the final-token hidden state as the sequence representation for a
#           two-way classification head.

import zipfile
import os
from pathlib import Path
import requests
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import pandas as pd
