#!/usr/bin/env python3
"""
build_exercises.py — construct the practice trees from the reference solutions.

For every file you are meant to fill in, two variants are produced, sharing the
SAME descriptive top-of-file comment:

  bare/          only the top comment + the necessary imports (you write
                 everything else, including the class/function headers)
  with-headers/  the top comment + imports + method/function headers with
                 `raise NotImplementedError` bodies (a first pass so you don't
                 get stuck)

Support files (tests, data, trainers, tokenizers, reference variants) are copied
unchanged into both variants so the with-headers tree stays runnable. The
solution clones are never modified.
"""
from __future__ import annotations
import os
import sys
import shutil
import py_compile

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from strip_code import strip_source, bare_source     # noqa: E402
from nb2py import convert as nb_convert              # noqa: E402

KSOL = os.path.join(EX, "karpathy-solutions")
RSOL = os.path.join(EX, "raschka-solutions")
K = os.path.join(EX, "karpathy")
R = os.path.join(EX, "raschka")
RREPO = os.path.join(RSOL, "LLMs-from-scratch")

IGNORE = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".ipynb_checkpoints",
                                ".DS_Store", "*.egg-info", ".pytest_cache")

def say(m): print(m)


def fresh_copy(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copytree(src, dst, ignore=IGNORE)


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def transform_fillable(path, names, topline, variant):
    src = read_text(path)
    if variant == "bare":
        out = bare_source(src, topline)
    else:
        out = topline.rstrip() + "\n\n\n" + strip_source(src, set(names) if names else None).lstrip("\n")
    write_text(path, out)


# --------------------------------------------------------------------------- #
# Top-of-file descriptions (what the file should do; inputs/outputs/behavior;
# deliberately NO implementation hints). Shared by both variants.
# --------------------------------------------------------------------------- #
T_ENGINE = """\
# engine.py -- a minimal scalar-valued automatic differentiation engine.
#
# Implement a `Value` type that wraps a single number and remembers how it was
# computed, so gradients can be propagated backwards through the whole
# expression.
#
# Input:    Python numbers combined with arithmetic operators.
# Output:   `Value` objects supporting + - * / ** , unary negation and a ReLU,
#           each exposing `.data` and `.grad`, plus a `.backward()`.
# Behavior: composing operations builds an expression graph; calling
#           `.backward()` on the final Value fills in `.grad` for every Value
#           that fed into it (reverse-mode autodiff)."""

T_NN = """\
# nn.py -- a tiny neural-network library built on the Value autograd engine.
#
# Implement the pieces of a multi-layer perceptron, expressed entirely in terms
# of `Value`s: a base module (exposing parameters() and zero_grad()), a single
# neuron, a layer of neurons, and an MLP (a stack of layers).
#
# Input:    a list of numbers / Values (one feature vector).
# Output:   a Value or list of Values (the forward pass), and parameters() must
#           return every trainable Value so an optimizer can update them.
# Behavior: calling a module runs a forward pass; weights are initialized
#           randomly when the module is constructed."""

T_MAKEMORE = """\
# makemore.py -- character-level language models over a list of words.
#
# Read a file of words (one per line, e.g. names.txt) and train a model to
# generate new, similar words one character at a time.
#
# Implement: the dataset (words <-> integer index sequences, with start/end
#   tokens), the model zoo -- Bigram, MLP, RNN/GRU, Transformer (and BoW
#   baselines), each mapping character-index sequences to next-character logits
#   and an optional loss -- and autoregressive sampling of new sequences.
# Input:    a words file; CLI flags select the model and hyperparameters.
# Output:   a trained checkpoint and freshly sampled words.
# Behavior: batch -> forward to logits + loss -> backprop -> optimizer step,
#           with periodic evaluation and sampling."""

T_MODEL = """\
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
#           projects to vocabulary logits."""

T_CH02 = """\
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
# Behavior: tokenize once, then index sliding windows for next-token prediction."""

T_CH03 = """\
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
#           the projection into independent subspaces and concatenates them."""

T_CH04 = """\
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
# Behavior: a forward pass embeds, runs the blocks, and projects to logits."""

T_CH05 = """\
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
#           at a time from the model's logits."""

T_CH06 = """\
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
#           two-way classification head."""

T_CH07 = """\
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
#           (and optionally prompt) positions from the loss."""

# nn-zero-to-hero lecture worksheets
W = {
 "micrograd/micrograd_lecture_first_half_roughly": """\
# Lecture: micrograd, part 1 -- backpropagation from scratch.
# Build a scalar autograd `Value` (its data, its grad, and the local backward of
# each operation) and see how reverse-mode differentiation walks the expression
# graph. Follow the lecture video and rebuild each cell yourself.""",
 "micrograd/micrograd_lecture_second_half_roughly": """\
# Lecture: micrograd, part 2 -- a neural net on top of Value.
# Use Value to build neurons / layers / an MLP, define a loss, and train by hand:
# forward, backward, nudge the parameters along their gradients, repeat.""",
 "makemore/makemore_part1_bigrams": """\
# Lecture: makemore, part 1 -- the bigram character language model.
# Count character bigrams from names.txt, turn the counts into probabilities and
# sample new names, then re-derive the same model as a 1-layer neural net trained
# with negative log-likelihood.""",
 "makemore/makemore_part2_mlp": """\
# Lecture: makemore, part 2 -- a multi-layer perceptron language model.
# Embed a context of characters and predict the next one with an MLP. Learn the
# training essentials: minibatches, learning-rate search, train/dev/test splits.""",
 "makemore/makemore_part3_bn": """\
# Lecture: makemore, part 3 -- activations, gradients, and BatchNorm.
# Diagnose a deeper network by inspecting the statistics of its activations and
# gradients, fix the initialization, and add BatchNorm to keep training stable.""",
 "makemore/makemore_part4_backprop": """\
# Lecture: makemore, part 4 -- becoming a backprop ninja.
# Manually backpropagate through the entire MLP + BatchNorm model with no
# autograd, deriving the gradient of every intermediate tensor by hand.""",
 "makemore/makemore_part5_cnn1": """\
# Lecture: makemore, part 5 -- a WaveNet-style hierarchical conv net.
# Grow the MLP into a deeper, tree-like model that fuses characters in stages,
# and practice using torch.nn containers cleanly.""",
}


def build_karpathy_variant(variant):
    dst_root = os.path.join(K, variant)

    # micrograd
    repo = os.path.join(dst_root, "micrograd")
    fresh_copy(os.path.join(KSOL, "micrograd"), repo)
    transform_fillable(os.path.join(repo, "micrograd", "engine.py"), None, T_ENGINE, variant)
    transform_fillable(os.path.join(repo, "micrograd", "nn.py"), None, T_NN, variant)
    for nb in ["demo.ipynb", "trace_graph.ipynb"]:
        p = os.path.join(repo, nb)
        write_text(p[:-6] + ".py", nb_convert(p, strip=False)); os.remove(p)

    # makemore
    repo = os.path.join(dst_root, "makemore")
    fresh_copy(os.path.join(KSOL, "makemore"), repo)
    transform_fillable(os.path.join(repo, "makemore.py"), MAKEMORE_MODELS, T_MAKEMORE, variant)

    # minGPT
    repo = os.path.join(dst_root, "minGPT")
    fresh_copy(os.path.join(KSOL, "minGPT"), repo)
    transform_fillable(os.path.join(repo, "mingpt", "model.py"), None, T_MODEL, variant)
    for nb in ["demo.ipynb", "generate.ipynb"]:
        p = os.path.join(repo, nb)
        write_text(p[:-6] + ".py", nb_convert(p, strip=False)); os.remove(p)

    # nn-zero-to-hero (lecture worksheets)
    repo = os.path.join(dst_root, "nn-zero-to-hero")
    fresh_copy(os.path.join(KSOL, "nn-zero-to-hero"), repo)
    for key, topline in W.items():
        nb = os.path.join(repo, "lectures", key + ".ipynb")
        full = nb_convert(nb, strip=False)
        if variant == "bare":
            code = bare_source(full, topline)
        else:
            code = topline.rstrip() + "\n\n\n" + nb_convert(nb, strip=True).lstrip("\n")
        write_text(nb[:-6] + ".py", code); os.remove(nb)
    shutil.copy(os.path.join(KSOL, "makemore", "names.txt"),
                os.path.join(repo, "lectures", "makemore", "names.txt"))


MAKEMORE_MODELS = ["NewGELU", "CausalSelfAttention", "Block", "Transformer",
                   "CausalBoW", "BoWBlock", "BoW", "RNNCell", "GRUCell", "RNN",
                   "MLP", "Bigram", "CharDataset", "generate"]

RASCHKA_STRIP = {
    "ch02.py": (["GPTDatasetV1", "create_dataloader_v1"], T_CH02),
    "ch03.py": (["SelfAttention_v1", "SelfAttention_v2", "CausalAttention",
                 "MultiHeadAttentionWrapper", "MultiHeadAttention"], T_CH03),
    "ch04.py": (["LayerNorm", "GELU", "FeedForward", "TransformerBlock",
                 "GPTModel", "generate_text_simple"], T_CH04),
    "ch05.py": (["generate", "train_model_simple", "evaluate_model",
                 "generate_and_print_sample", "calc_loss_batch", "calc_loss_loader",
                 "text_to_token_ids", "token_ids_to_text", "load_weights_into_gpt"], T_CH05),
    "ch06.py": (["SpamDataset", "calc_accuracy_loader", "calc_loss_batch",
                 "calc_loss_loader", "evaluate_model", "train_classifier_simple",
                 "classify_review"], T_CH06),
    "ch07.py": (["format_input", "InstructionDataset", "custom_collate_fn"], T_CH07),
}

RASCHKA_PYPROJECT = '''\
# Lightweight, editable install of the (stripped) llms_from_scratch package.
#   cd raschka/<variant> && pip install -e .
# Heavy extras (tensorflow for ch05 GPT-2 weight loading) are intentionally
# omitted -- install them yourself if you reach chapter 5's weight download.
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "llms-from-scratch-exercises"
version = "0.0.1"
description = "Stripped exercise skeleton of Sebastian Raschka's LLMs-from-scratch"
requires-python = ">=3.10,<3.13"
dependencies = [
  "torch>=2.2.2", "tiktoken>=0.5.1", "numpy>=1.26", "pandas>=2.2.1",
  "matplotlib>=3.7.1", "tqdm>=4.66.1", "pytest>=8.0",
  "psutil>=5.9.5", "requests>=2.31",
]

[tool.setuptools]
package-dir = {"" = "pkg"}

[tool.setuptools.packages.find]
where = ["pkg"]
'''

RASCHKA_CONFTEST = '''\
"""Make `import llms_from_scratch` work when running pytest without installing.
(You can still `pip install -e .` instead.)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pkg"))
'''


def build_raschka_variant(variant):
    dst_root = os.path.join(R, variant)
    fresh_copy(os.path.join(RREPO, "pkg"), os.path.join(dst_root, "pkg"))
    lfs = os.path.join(dst_root, "pkg", "llms_from_scratch")
    for fname, (names, topline) in RASCHKA_STRIP.items():
        transform_fillable(os.path.join(lfs, fname), names, topline, variant)
    write_text(os.path.join(dst_root, "pyproject.toml"), RASCHKA_PYPROJECT)
    write_text(os.path.join(dst_root, "conftest.py"), RASCHKA_CONFTEST)
    verdict = os.path.join(RREPO, "ch02", "01_main-chapter-code", "the-verdict.txt")
    if os.path.exists(verdict):
        write_text(os.path.join(dst_root, "data", "the-verdict.txt"), read_text(verdict))


def add_solution_conversions():
    """Add a full .py next to every Karpathy solution notebook (kept alongside
    the .ipynb) so the reference answers are available as plain Python too."""
    for repo in ("micrograd", "makemore", "minGPT", "nn-zero-to-hero"):
        base = os.path.join(KSOL, repo)
        for root, _, files in os.walk(base):
            if ".git" in root:
                continue
            for fn in sorted(files):
                if fn.endswith(".ipynb"):
                    p = os.path.join(root, fn)
                    write_text(p[:-6] + ".py", nb_convert(p, strip=False))
    src = os.path.join(KSOL, "makemore", "names.txt")
    dst = os.path.join(KSOL, "nn-zero-to-hero", "lectures", "makemore", "names.txt")
    if os.path.exists(src):
        shutil.copy(src, dst)


def compile_check():
    say("\n=== COMPILE CHECK ===")
    bad = 0
    for base in (K, R):
        for root, _, files in os.walk(base):
            if "__pycache__" in root:
                continue
            for fn in files:
                if fn.endswith(".py"):
                    p = os.path.join(root, fn)
                    try:
                        py_compile.compile(p, doraise=True)
                    except py_compile.PyCompileError as e:
                        bad += 1
                        say(f"  !! {os.path.relpath(p, EX)}: {e}")
    say(f"  {'PASSED' if bad == 0 else f'FAILED ({bad})'}")
    return bad == 0


if __name__ == "__main__":
    for d in (K, R):
        if os.path.exists(d):
            shutil.rmtree(d)
    for variant in ("bare", "with-headers"):
        say(f"\n=== building {variant} ===")
        build_karpathy_variant(variant)
        build_raschka_variant(variant)
    add_solution_conversions()
    ok = compile_check()
    say("\nDONE." if ok else "\nDONE WITH ERRORS.")
    sys.exit(0 if ok else 1)
