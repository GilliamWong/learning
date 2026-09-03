# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "tensorboard",
#     "torch>=2.2",
# ]
# ///
"""Print and validate the full Transformer: embeddings, logits, loss, sampling.

Transformer wraps the block stack with the token and position embeddings, the
final LayerNorm, the language-model head, and the loss. The cases below check
that wrapping and the contract the training loop and generate() rely on.

Run check_attention.py and check_block.py first -- this model contains both, so
a failure further down will surface here with a vaguer message.

Run from the repository root with:

    uv run exercises/karpathy/with-headers/makemore/check_transformer.py
"""

from __future__ import annotations

import math
import sys
import traceback

import torch
import torch.nn as nn
from torch.nn import functional as F

from makemore import Block, ModelConfig, Transformer, generate


BLOCK_SIZE = 8
VOCAB_SIZE = 27
N_LAYER = 2
N_EMBD = 12
N_HEAD = 3


def make_config(**overrides) -> ModelConfig:
    kwargs = dict(
        block_size=BLOCK_SIZE,
        vocab_size=VOCAB_SIZE,
        n_layer=N_LAYER,
        n_embd=N_EMBD,
        n_embd2=N_EMBD,
        n_head=N_HEAD,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def print_heading(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def print_exception(error: BaseException) -> None:
    print(f"ERROR TYPE: {type(error).__name__}")
    print(f"ERROR MESSAGE: {error}")
    print("TRACEBACK:")
    traceback.print_exc(file=sys.stdout)


def result_line(passed: bool) -> None:
    print(f"\nCASE RESULT: {'PASS' if passed else 'FAIL'}")


def random_batch(batch: int = 3, seq_len: int = BLOCK_SIZE):
    """An (idx, targets) pair shaped like CharDataset.__getitem__ produces."""
    idx = torch.randint(0, VOCAB_SIZE, (batch, seq_len))
    targets = torch.randint(0, VOCAB_SIZE, (batch, seq_len))
    # CharDataset pads the tail of short words with -1 so the loss skips them.
    # Pad a different amount on each row, leaving at least one row unpadded.
    for row in range(1, batch):
        padding = row % min(seq_len, 4)
        if padding:
            targets[row, -padding:] = -1
    return idx, targets


def check_structure(model: Transformer) -> bool:
    print_heading("Module and parameter inventory")
    print("submodules:")
    for name, module in model.named_modules():
        if name and "." not in name.replace("transformer.", "", 1):
            print(f"  {name:<28} {type(module).__name__}")

    modules = [m for name, m in model.named_modules() if name]
    embeddings = [m for m in modules if isinstance(m, nn.Embedding)]
    blocks = [m for m in modules if isinstance(m, Block)]
    layer_norms = [m for m in modules if isinstance(m, nn.LayerNorm)]
    heads = [
        m
        for m in modules
        if isinstance(m, nn.Linear) and m.out_features == VOCAB_SIZE
    ]

    token_tables = [m for m in embeddings if m.num_embeddings == VOCAB_SIZE]
    position_tables = [m for m in embeddings if m.num_embeddings == BLOCK_SIZE]

    print(f"\nEmbedding tables:            {len(embeddings)} (expected 2)")
    print(f"  sized vocab_size={VOCAB_SIZE}:      {len(token_tables)} (expected 1, wte)")
    print(f"  sized block_size={BLOCK_SIZE}:       {len(position_tables)} (expected 1, wpe)")
    print(f"Block modules:               {len(blocks)} (expected n_layer = {N_LAYER})")
    print(f"LayerNorm modules:           {len(layer_norms)} "
          f"(expected {2 * N_LAYER + 1} = 2 per block + 1 final)")
    print(f"Linear(-> vocab_size):       {len(heads)} (expected 1, lm_head)")

    if not position_tables:
        print("no position embedding table found. Attention is permutation invariant,")
        print("so without wpe the model cannot tell 'ab' from 'ba'.")
    if len(layer_norms) == 2 * N_LAYER:
        print("there are exactly 2 per block and none left over -- GPT-2 has one more")
        print("LayerNorm after the stack, before lm_head.")

    block_params = 12 * N_EMBD * N_EMBD + 13 * N_EMBD
    base = (
        VOCAB_SIZE * N_EMBD          # wte
        + BLOCK_SIZE * N_EMBD        # wpe
        + N_LAYER * block_params     # blocks
        + 2 * N_EMBD                 # ln_f
    )
    without_head_bias = base + N_EMBD * VOCAB_SIZE
    with_head_bias = without_head_bias + VOCAB_SIZE
    total = sum(p.numel() for p in model.parameters())
    print(f"\nparameter total: {total}")
    print(f"expected {without_head_bias} with lm_head bias=False (what GPT-2 uses)")
    print(f"expected {with_head_bias} with lm_head bias=True")

    structure_ok = (
        len(token_tables) == 1
        and len(position_tables) == 1
        and len(blocks) == N_LAYER
        and len(layer_norms) == 2 * N_LAYER + 1
        and len(heads) == 1
    )
    count_ok = total in (without_head_bias, with_head_bias)

    passed = structure_ok and count_ok
    result_line(passed)
    return passed


def check_get_block_size(model: Transformer) -> bool:
    print_heading("get_block_size()")
    print("generate() calls this every step to crop the context, so it has to report")
    print("the same block_size the position table was built with.")
    try:
        reported = model.get_block_size()
        print(f"  returned: {reported!r}")
        print(f"  config:   {BLOCK_SIZE}")
        passed = reported == BLOCK_SIZE
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def check_forward_without_targets(model: Transformer) -> bool:
    print_heading("forward(idx) with no targets")
    print("The training loop calls the model both ways, and generate() always calls")
    print("it without targets, so it must return a (logits, None) pair either way.")

    idx, _ = random_batch()
    try:
        output = model(idx)
        print(f"  returned type: {type(output).__name__}")
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("forward must return a 2-tuple of (logits, loss)")

        logits, loss = output
        expected_shape = (idx.shape[0], idx.shape[1], VOCAB_SIZE)
        print(f"  logits shape: {tuple(logits.shape)} (expected {expected_shape})")
        print(f"  logits dtype: {logits.dtype}")
        print(f"  loss:         {loss!r} (expected None)")

        shape_ok = tuple(logits.shape) == expected_shape
        loss_ok = loss is None
        finite_ok = bool(torch.isfinite(logits).all())
        print(f"  all logits finite: {finite_ok}")
        if not shape_ok:
            print("  logits must be (B, T, vocab_size): one distribution per position,")
            print("  not just one for the final position -- the loss needs them all.")

        passed = shape_ok and loss_ok and finite_ok
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def check_loss_matches_cross_entropy(model: Transformer) -> bool:
    print_heading("Loss equals cross entropy over the non-ignored positions")
    print("CharDataset pads short words with -1 in Y, so the loss must pass")
    print("ignore_index=-1. Without it those positions are treated as class -1 and")
    print("the loss is wrong (or errors outright).")

    idx, targets = random_batch()
    print(f"  targets contain {int((targets == -1).sum())} ignored positions out of "
          f"{targets.numel()}")
    try:
        logits, loss = model(idx, targets)
        if loss is None:
            raise AssertionError("forward returned loss=None even though targets were given")

        print(f"  loss shape: {tuple(loss.shape)} (expected () -- a scalar)")
        print(f"  loss value: {loss.item():.9f}")

        flattened = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), ignore_index=-1
        )
        mask = targets != -1
        masked = F.cross_entropy(logits[mask], targets[mask])
        print(f"  reference with ignore_index=-1:        {flattened.item():.9f}")
        print(f"  reference over kept positions only:    {masked.item():.9f}")

        scalar_ok = loss.shape == torch.Size([])
        matches = abs(loss.item() - flattened.item()) < 1e-5
        ignores = abs(loss.item() - masked.item()) < 1e-5
        print(f"\n  loss is a scalar: {scalar_ok}")
        print(f"  matches ignore_index reference: {matches}")
        print(f"  ignored positions excluded:     {ignores}")
        if matches and not ignores:
            print("  (these two references agree for correct code; if only one matches,")
            print("  re-read how the -1 positions are being averaged)")
        if not matches:
            print("  check the flatten: cross_entropy wants (B*T, vocab) and (B*T,),")
            print("  the same reshape you used in RNN.forward.")

        passed = scalar_ok and matches and ignores
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def check_initial_loss(model: Transformer) -> bool:
    print_heading("Loss at initialisation is about ln(vocab_size)")
    print("An untrained model should be roughly uniform over the vocabulary, so the")
    print("loss starts near ln(vocab_size). A much larger value usually means the")
    print("initialisation is too wide; a much smaller one means something is leaking")
    print("the answer.")

    idx, targets = random_batch(batch=64)
    expected = math.log(VOCAB_SIZE)
    try:
        with torch.no_grad():
            _, loss = model(idx, targets)
        print(f"  loss:            {loss.item():.6f}")
        print(f"  ln(vocab_size):  {expected:.6f}")
        print(f"  difference:      {abs(loss.item() - expected):.6f} (want < 1.0)")

        passed = abs(loss.item() - expected) < 1.0
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def check_position_embedding_used(model: Transformer) -> bool:
    print_heading("The position embedding actually changes the output")
    print("Feed the same token at every position. Attention and the MLP are both")
    print("position-blind, so if wpe is missing or never added, every row of logits")
    print("comes out identical. They must differ.")

    idx = torch.full((1, BLOCK_SIZE), 5, dtype=torch.long)
    try:
        with torch.no_grad():
            logits, _ = model(idx)
        for t in range(BLOCK_SIZE):
            drift = (logits[0, t] - logits[0, 0]).abs().max().item()
            print(f"  position {t} differs from position 0 by: {drift:.6e}")
        spread = (logits - logits[:, :1, :]).abs().max().item()
        print(f"\n  maximum spread across positions: {spread:.6e} (want > 0)")
        if spread == 0.0:
            print("  identical rows: add wpe = nn.Embedding(block_size, n_embd) and sum")
            print("  it with the token embedding before the block stack.")

        passed = spread > 1e-6
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def check_causality(model: Transformer) -> bool:
    print_heading("Changing a token only affects that position and later ones")
    print("This is the end-to-end version of the causal check. If an earlier")
    print("position moves, the model is peeking at the future and its training loss")
    print("will look impossibly good.")

    idx, _ = random_batch(batch=1)
    try:
        with torch.no_grad():
            baseline, _ = model(idx)

        results = []
        for t in range(1, BLOCK_SIZE):
            changed = idx.clone()
            changed[0, t] = (changed[0, t] + 7) % VOCAB_SIZE
            with torch.no_grad():
                perturbed, _ = model(changed)
            before = (perturbed[0, :t] - baseline[0, :t]).abs().max().item()
            at_and_after = (perturbed[0, t:] - baseline[0, t:]).abs().max().item()
            ok = before == 0.0 and at_and_after > 0.0
            results.append(ok)
            print(f"  changed token {t}: positions < {t} moved by {before:.3e} "
                  f"(want 0), positions >= {t} moved by {at_and_after:.3e} (want > 0)")

        passed = all(results)
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def check_variable_sequence_length(model: Transformer) -> bool:
    print_heading("Variable sequence length T = 1 .. block_size")
    print("Sampling starts from a single START token, so T=1 has to work.")
    results = []
    for seq_len in range(1, BLOCK_SIZE + 1):
        idx = torch.randint(0, VOCAB_SIZE, (2, seq_len))
        try:
            with torch.no_grad():
                logits, _ = model(idx)
            ok = tuple(logits.shape) == (2, seq_len, VOCAB_SIZE)
            print(f"  T={seq_len:<3} logits {str(tuple(logits.shape)):<18} "
                  f"{'ok' if ok else 'WRONG'}")
        except Exception as error:
            ok = False
            print(f"  T={seq_len:<3} raised {type(error).__name__}: {error}")
        results.append(ok)

    passed = all(results)
    result_line(passed)
    return passed


def check_too_long_sequence(model: Transformer) -> bool:
    print_heading("A sequence longer than block_size is rejected")
    print(f"Feeding T={BLOCK_SIZE + 1} when block_size={BLOCK_SIZE}. There is no")
    print("position embedding for that slot, so this should fail loudly rather than")
    print("read past the end of the table.")
    idx = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE + 1))
    try:
        with torch.no_grad():
            model(idx)
    except Exception as error:
        print(f"  raised {type(error).__name__}: {error}")
        print("\nCASE RESULT: PASS")
        return True
    print("  no error raised -- minGPT asserts the sequence fits the block size.")
    print("\nCASE RESULT: FAIL")
    return False


def check_gradients(model: Transformer) -> bool:
    print_heading("Every parameter receives a gradient")
    print("Run at T=block_size so every row of the position table is touched. A")
    print("parameter with no gradient was created and then never used.")

    model.zero_grad(set_to_none=True)
    idx, targets = random_batch(batch=4)
    try:
        _, loss = model(idx, targets)
        loss.backward()
        results = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                print(f"  {name:<40} NO GRADIENT")
                results.append(False)
                continue
            magnitude = parameter.grad.abs().sum().item()
            ok = magnitude > 0.0
            print(f"  {name:<40} sum |grad| = {magnitude:.6e} {'ok' if ok else 'ZERO'}")
            results.append(ok)
        model.zero_grad(set_to_none=True)

        passed = bool(results) and all(results)
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def check_generate(model: Transformer) -> bool:
    print_heading("End to end through makemore's own generate()")
    print("This is the real sampling path: it starts at T=1 and grows past")
    print("block_size, where it begins cropping the context. Anything that assumed a")
    print("fixed T shows up here even when training would have worked.")

    start = torch.zeros(4, 1, dtype=torch.long)
    steps = BLOCK_SIZE + 3
    try:
        sampled = generate(model, start, steps, do_sample=True, top_k=5)
        print(f"  input shape:  {tuple(start.shape)}")
        print(f"  output shape: {tuple(sampled.shape)} "
              f"(expected {(4, 1 + steps)})")
        print(f"  output dtype: {sampled.dtype} (expected torch.int64)")
        print(f"  sampled indices:\n{sampled}")

        shape_ok = tuple(sampled.shape) == (4, 1 + steps)
        dtype_ok = sampled.dtype == torch.int64
        range_ok = bool(((sampled >= 0) & (sampled < VOCAB_SIZE)).all())
        print(f"\n  shape ok: {shape_ok}")
        print(f"  dtype ok: {dtype_ok}")
        print(f"  all indices within [0, vocab_size): {range_ok}")

        greedy = generate(model, start, 4, do_sample=False)
        identical = bool((greedy[0] == greedy[1]).all())
        print(f"  greedy decoding gives every row the same continuation: {identical} "
              f"(expected True, all rows start from the same token)")

        passed = shape_ok and dtype_ok and range_ok and identical
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def check_determinism(model: Transformer) -> bool:
    print_heading("Deterministic, and identical in train and eval mode")
    idx, targets = random_batch()
    try:
        model.train()
        with torch.no_grad():
            first, first_loss = model(idx, targets)
            second, _ = model(idx, targets)
            model.eval()
            third, third_loss = model(idx, targets)
        model.train()

        repeat_delta = (first - second).abs().max().item()
        mode_delta = (first - third).abs().max().item()
        print(f"  two calls in train mode differ by: {repeat_delta:.3e} (want 0)")
        print(f"  train vs eval mode differs by:     {mode_delta:.3e} (want 0)")
        print(f"  train loss {first_loss.item():.9f} vs eval loss {third_loss.item():.9f}")

        passed = repeat_delta == 0.0 and mode_delta == 0.0
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def main() -> int:
    torch.set_printoptions(precision=6, linewidth=160, threshold=10_000)
    torch.manual_seed(3407)

    print_heading("Transformer diagnostic")
    print(f"Python:  {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"config:  block_size={BLOCK_SIZE}, vocab_size={VOCAB_SIZE}, "
          f"n_layer={N_LAYER}, n_embd={N_EMBD}, n_head={N_HEAD}")
    print("Run check_attention.py and check_block.py first -- both are inside this")
    print("model, and their failures resurface here less legibly.")

    config = make_config()
    try:
        model = Transformer(config)
    except Exception as error:
        print_heading("Construction")
        print("Transformer(config) failed, so no other case can run.")
        print_exception(error)
        print("\nOVERALL RESULT: FAIL")
        return 1

    results = []
    results.append(check_structure(model))
    results.append(check_get_block_size(model))
    results.append(check_forward_without_targets(model))
    results.append(check_loss_matches_cross_entropy(model))
    results.append(check_initial_loss(model))
    results.append(check_position_embedding_used(model))
    results.append(check_causality(model))
    results.append(check_variable_sequence_length(model))
    results.append(check_too_long_sequence(model))
    results.append(check_gradients(model))
    results.append(check_generate(model))
    results.append(check_determinism(model))

    print_heading("Summary")
    ran = [r for r in results if r is not None]
    skipped = len(results) - len(ran)
    print(f"passed: {sum(ran)}/{len(ran)}" + (f" ({skipped} skipped)" if skipped else ""))
    if all(ran):
        print("OVERALL RESULT: PASS")
        return 0

    print("OVERALL RESULT: FAIL")
    print("The command exits with status 1 so terminal tooling can detect the failure.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
