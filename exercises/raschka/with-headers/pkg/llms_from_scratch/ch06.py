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


# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import zipfile
import os
from pathlib import Path

import requests
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import pandas as pd


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


def create_balanced_dataset(df):

    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        # TODO: implement SpamDataset.__init__
        raise NotImplementedError

    def __getitem__(self, index):
        # TODO: implement SpamDataset.__getitem__
        raise NotImplementedError

    def __len__(self):
        # TODO: implement SpamDataset.__len__
        raise NotImplementedError

    def _longest_encoded_length(self):
        # TODO: implement SpamDataset._longest_encoded_length
        raise NotImplementedError
        # Note: A more pythonic version to implement this method
        # is the following, which is also used in the next chapter:
        # return max(len(encoded_text) for encoded_text in self.encoded_texts)


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    # TODO: implement calc_accuracy_loader
    raise NotImplementedError


def calc_loss_batch(input_batch, target_batch, model, device):
    # TODO: implement calc_loss_batch
    raise NotImplementedError


def calc_loss_loader(data_loader, model, device, num_batches=None):
    # TODO: implement calc_loss_loader
    raise NotImplementedError


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # TODO: implement evaluate_model
    raise NotImplementedError


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # Initialize lists to track losses and examples seen
    # TODO: implement train_classifier_simple
    raise NotImplementedError


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    # TODO: implement classify_review
    raise NotImplementedError
