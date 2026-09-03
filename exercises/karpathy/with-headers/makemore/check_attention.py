# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "tensorboard",
#     "torch>=2.2",
# ]
# ///
"""Print and validate CausalSelfAttention structure, causality, and values.

Unlike NewGELU there is no fixed reference to compare against: the module owns
random weights, so most checks below are properties that any correct causal
self-attention must satisfy no matter how it is written. One case does build an
exact reference, by reusing the module's own Linear layers.

Run from the repository root with:

    uv run exercises/karpathy/with-headers/makemore/check_attention.py
"""

from __future__ import annotations

import sys
import traceback

import torch
import torch.nn as nn
from torch.nn import functional as F

from makemore import CausalSelfAttention, ModelConfig


BLOCK_SIZE = 8
N_EMBD = 12
N_HEAD = 3
HEAD_SIZE = N_EMBD // N_HEAD


def make_config(**overrides) -> ModelConfig:
    kwargs = dict(
        block_size=BLOCK_SIZE,
        vocab_size=27,
        n_layer=2,
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


def check_inventory(model: CausalSelfAttention) -> bool:
    print_heading("Parameter and buffer inventory")
    print("named parameters:")
    total = 0
    parameter_shapes = []
    for name, parameter in model.named_parameters():
        print(f"  {name:<24} {str(tuple(parameter.shape)):<16} {parameter.numel()} values")
        total += parameter.numel()
        parameter_shapes.append((name, tuple(parameter.shape)))
    if total == 0:
        print("  (none)")

    print("\nnamed buffers:")
    buffer_names = []
    for name, buffer in model.named_buffers():
        print(f"  {name:<24} {str(tuple(buffer.shape)):<16} dtype={buffer.dtype}")
        buffer_names.append(name)
    if not buffer_names:
        print("  (none)")

    with_bias = 4 * N_EMBD * N_EMBD + 4 * N_EMBD
    without_bias = 4 * N_EMBD * N_EMBD
    print(f"\nparameter total: {total}")
    print(f"expected 4*C*C + 4*C = {with_bias} (Linear layers with bias)")
    print(f"expected 4*C*C       = {without_bias} (Linear layers with bias=False)")
    count_ok = total in (with_bias, without_bias)
    if not count_ok:
        print("parameter count matches neither expectation -- check for an extra or")
        print("missing Linear, or for the causal mask being stored as a Parameter.")

    mask_as_parameter = [
        name
        for name, shape in parameter_shapes
        if BLOCK_SIZE in shape and N_EMBD not in shape
    ]
    if mask_as_parameter:
        print(f"the causal mask looks like a Parameter: {mask_as_parameter}")
        print("it should be a buffer (register_buffer) so it is not trained.")
    mask_ok = not mask_as_parameter

    if not buffer_names:
        print("\nnote: no buffers registered. That is correct if you call")
        print("F.scaled_dot_product_attention(..., is_causal=True), and a bug if you")
        print("built a tril mask and stored it as a plain attribute -- a plain")
        print("attribute does not follow model.to(device).")

    passed = count_ok and mask_ok
    result_line(passed)
    return passed


def check_forward(model: CausalSelfAttention, name: str, x: torch.Tensor) -> bool:
    print_heading(name)
    print(f"input shape:  {tuple(x.shape)}")
    print(f"input dtype:  {x.dtype}")
    print(f"input device: {x.device}")

    try:
        actual = model(x)

        if not isinstance(actual, torch.Tensor):
            raise TypeError(
                f"CausalSelfAttention returned {type(actual).__name__}, not torch.Tensor"
            )

        print(f"output shape:  {tuple(actual.shape)}")
        print(f"output dtype:  {actual.dtype}")
        print(f"output device: {actual.device}")
        print(f"output:\n{actual}")

        shape_ok = actual.shape == x.shape
        dtype_ok = actual.dtype == x.dtype
        device_ok = actual.device == x.device
        finite_ok = bool(torch.isfinite(actual).all())

        print(f"\nshape preserved (B, T, n_embd in -> out): {shape_ok}")
        print(f"dtype preserved: {dtype_ok}")
        print(f"device preserved: {device_ok}")
        print(f"all values finite: {finite_ok}")

        if not shape_ok:
            print("the residual add in Block is x + attn(ln_1(x)), so the output must")
            print("have exactly the input shape. Check the output projection and the")
            print("transpose/view that merges the heads back together.")
        if not finite_ok:
            print("non-finite values usually mean an entire attention row was filled")
            print("with -inf before the softmax, which produces nan.")

        passed = shape_ok and dtype_ok and device_ok and finite_ok
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def check_variable_sequence_length(model: CausalSelfAttention) -> bool:
    print_heading("Variable sequence length T = 1 .. block_size")
    print("generate() feeds T=1 first and grows the context one token at a time, so")
    print("a mask hardcoded to block_size instead of sliced to [:T, :T] trains fine")
    print("and then breaks during sampling.")

    results = []
    for seq_len in range(1, BLOCK_SIZE + 1):
        x = torch.randn(2, seq_len, N_EMBD)
        try:
            y = model(x)
            ok = tuple(y.shape) == (2, seq_len, N_EMBD) and bool(torch.isfinite(y).all())
            print(f"  T={seq_len:<3} output {str(tuple(y.shape)):<16} {'ok' if ok else 'WRONG'}")
        except Exception as error:
            ok = False
            print(f"  T={seq_len:<3} raised {type(error).__name__}: {error}")
        results.append(ok)

    passed = all(results)
    result_line(passed)
    return passed


def check_causality(model: CausalSelfAttention) -> bool:
    print_heading("Causality via autograd (the important one)")
    print("d output[t] / d input[t'] must be exactly zero for every future t' > t.")
    print("A nonzero future gradient means the mask is missing, inverted, or applied")
    print("after the softmax instead of before it.")

    seq_len = BLOCK_SIZE
    x = torch.randn(2, seq_len, N_EMBD, requires_grad=True)

    future_ok = []
    past_ok = []
    try:
        for t in range(seq_len):
            if x.grad is not None:
                x.grad = None
            y = model(x)
            y[:, t, :].sum().backward()
            gradient = x.grad
            assert gradient is not None

            future = gradient[:, t + 1 :, :].abs().max().item() if t + 1 < seq_len else 0.0
            past = gradient[:, : t + 1, :].abs().max().item()
            leak = future == 0.0
            flow = past > 0.0
            future_ok.append(leak)
            past_ok.append(flow)
            print(
                f"  output position {t}: max |grad| from future = {future:.3e} "
                f"({'ok' if leak else 'LEAK'}), from past+self = {past:.3e} "
                f"({'ok' if flow else 'DEAD'})"
            )

        if not all(future_ok):
            print("\ninformation is leaking backwards from future tokens.")
        if not all(past_ok):
            print("\nsome outputs do not depend on their own position, which usually")
            print("means the mask is shifted by one -- strictly lower triangular")
            print("instead of lower triangular including the diagonal.")

        passed = all(future_ok) and all(past_ok)
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def check_batch_independence(model: CausalSelfAttention) -> bool:
    print_heading("Batch independence")
    print("Changing one sequence in the batch must not change any other sequence.")
    print("A reshape that folds B into T shows up here.")

    x = torch.randn(3, BLOCK_SIZE, N_EMBD)
    try:
        y = model(x)
        perturbed = x.clone()
        perturbed[1] = torch.randn(BLOCK_SIZE, N_EMBD)
        y_perturbed = model(perturbed)

        row0 = (y[0] - y_perturbed[0]).abs().max().item()
        row2 = (y[2] - y_perturbed[2]).abs().max().item()
        row1 = (y[1] - y_perturbed[1]).abs().max().item()
        print(f"  untouched batch row 0 changed by: {row0:.3e} (want 0)")
        print(f"  untouched batch row 2 changed by: {row2:.3e} (want 0)")
        print(f"  perturbed batch row 1 changed by: {row1:.3e} (want > 0)")

        passed = row0 == 0.0 and row2 == 0.0 and row1 > 0.0
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def check_uniform_input(model: CausalSelfAttention) -> bool:
    print_heading("Uniform input: attention weights must sum to one")
    print("Feed the same vector at every position. Attention knows nothing about")
    print("position, so every output row must come out identical: a weighted average")
    print("of identical values is that value, however many are averaged.")
    print("Rows that drift apart with t mean the weights are not normalised -- a sum")
    print("instead of a softmax, or a mask applied after the softmax.")

    row = torch.randn(1, 1, N_EMBD)
    x = row.expand(2, BLOCK_SIZE, N_EMBD).contiguous()
    try:
        y = model(x)
        for t in range(BLOCK_SIZE):
            drift = (y[0, t] - y[0, 0]).abs().max().item()
            print(f"  position {t} differs from position 0 by: {drift:.3e}")
        spread = (y - y[:, :1, :]).abs().max().item()
        print(f"\nmaximum spread across positions: {spread:.3e}")

        passed = spread < 1e-5
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def find_projections(model: CausalSelfAttention):
    """Best effort: locate the q/k/v projection(s) and the output projection."""
    linears = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    fused = [
        (n, m) for n, m in linears if m.in_features == N_EMBD and m.out_features == 3 * N_EMBD
    ]
    square = [
        (n, m) for n, m in linears if m.in_features == N_EMBD and m.out_features == N_EMBD
    ]

    if len(fused) == 1 and len(square) == 1:
        return {
            "layout": "fused",
            "qkv": fused[0][1],
            "proj": square[0][1],
            "detail": (
                f"fused qkv '{fused[0][0]}', output projection '{square[0][0]}'"
            ),
        }

    if not fused and len(square) == 4:
        keywords = ("proj", "out", "merge", "combine")
        named = [(n, m) for n, m in square if any(k in n.lower() for k in keywords)]
        if len(named) == 1:
            proj_name, proj = named[0]
            rest = [(n, m) for n, m in square if n != proj_name]
            return {
                "layout": "separate",
                "q": rest[0][1],
                "k": rest[1][1],
                "v": rest[2][1],
                "proj": proj,
                "detail": (
                    f"separate q='{rest[0][0]}', k='{rest[1][0]}', v='{rest[2][0]}', "
                    f"output projection '{proj_name}' -- q/k/v order guessed from "
                    "definition order"
                ),
            }
    return None


def reference_attention(layout, x: torch.Tensor) -> torch.Tensor:
    batch, seq_len, channels = x.shape
    if layout["layout"] == "fused":
        query, key, value = layout["qkv"](x).split(channels, dim=2)
    else:
        query, key, value = layout["q"](x), layout["k"](x), layout["v"](x)

    def split_heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch, seq_len, N_HEAD, HEAD_SIZE).transpose(1, 2)

    attended = F.scaled_dot_product_attention(
        split_heads(query), split_heads(key), split_heads(value), is_causal=True
    )
    merged = attended.transpose(1, 2).contiguous().view(batch, seq_len, channels)
    return layout["proj"](merged)


def check_against_reference(model: CausalSelfAttention):
    print_heading("Exact values against F.scaled_dot_product_attention")
    layout = find_projections(model)
    if layout is None:
        print("Could not identify the q/k/v and output Linear layers from their shapes")
        print("and names, so this exact comparison is skipped. Every other case still")
        print("ran. The Linear layers found were:")
        found = False
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"  {name}: Linear({module.in_features} -> {module.out_features})")
                found = True
        if not found:
            print("  (none)")
        print("\nCASE RESULT: SKIP")
        return None

    print(f"detected layout: {layout['detail']}")
    print("The reference reuses your own weights, so any difference is in the")
    print("attention maths itself: the 1/sqrt(head_size) scaling, the head")
    print("split/merge, or the mask.")

    x = torch.randn(2, BLOCK_SIZE, N_EMBD)
    try:
        with torch.no_grad():
            expected = reference_attention(layout, x)
            actual = model(x)

        print(f"\nexpected output:\n{expected}")
        print(f"\nactual output:\n{actual}")
        error = (actual - expected).abs()
        print(f"\nabsolute error:\n{error}")
        print(f"maximum absolute error: {error.max().item():.9g}")

        try:
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
            print("value comparison: PASS (atol=1e-05, rtol=0.0001)")
            print("\nCASE RESULT: PASS")
            return True
        except AssertionError as failure:
            print("value comparison: FAIL (atol=1e-05, rtol=0.0001)")
            print(failure)
            print("\nCommon causes, in the order worth checking:")
            print("  - forgot to divide the scores by sqrt(head_size)")
            print("  - split heads with view(B, nh, T, hs) instead of")
            print("    view(B, T, nh, hs).transpose(1, 2)")
            print("  - merged heads without .transpose(1, 2).contiguous() first")
            print("  - softmax over the wrong dimension (must be the key axis, -1)")
            print("  - q/k/v read out of the fused projection in a different order")
            print("\nCASE RESULT: FAIL")
            return False
    except Exception as failure:
        print("\nCASE RESULT: ERROR")
        print_exception(failure)
        return False


def check_head_divisibility() -> bool:
    print_heading("n_embd not divisible by n_head should be rejected")
    print("Building with n_embd=13, n_head=3. minGPT asserts on this in __init__;")
    print("without the assert you get a confusing reshape error much later.")
    config = make_config(n_embd=13, n_head=3)
    try:
        model = CausalSelfAttention(config)
        model(torch.randn(2, 4, 13))
    except Exception as error:
        print(f"raised {type(error).__name__}: {error}")
        print("\nCASE RESULT: PASS")
        return True
    print("no error raised -- add `assert config.n_embd % config.n_head == 0` to")
    print("__init__ so this fails loudly and early.")
    print("\nCASE RESULT: FAIL")
    return False


def check_determinism(model: CausalSelfAttention) -> bool:
    print_heading("Deterministic, and identical in train and eval mode")
    print("This file removed dropout, so there should be no randomness at all.")
    x = torch.randn(2, BLOCK_SIZE, N_EMBD)
    try:
        model.train()
        first = model(x)
        second = model(x)
        model.eval()
        third = model(x)
        model.train()

        repeat_delta = (first - second).abs().max().item()
        mode_delta = (first - third).abs().max().item()
        print(f"  two calls in train mode differ by: {repeat_delta:.3e} (want 0)")
        print(f"  train vs eval mode differs by:     {mode_delta:.3e} (want 0)")

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

    print_heading("CausalSelfAttention diagnostic")
    print(f"Python:  {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(
        f"config:  block_size={BLOCK_SIZE}, n_embd={N_EMBD}, n_head={N_HEAD}, "
        f"head_size={HEAD_SIZE}"
    )

    config = make_config()
    try:
        model = CausalSelfAttention(config)
    except Exception as error:
        print_heading("Construction")
        print("CausalSelfAttention(config) failed, so no other case can run.")
        print_exception(error)
        print("\nOVERALL RESULT: FAIL")
        return 1

    results = []
    results.append(check_inventory(model))
    results.append(
        check_forward(
            model,
            "Forward with shape (B=2, T=block_size=8, C=12)",
            torch.randn(2, BLOCK_SIZE, N_EMBD),
        )
    )
    results.append(
        check_forward(
            model,
            "Forward with one sequence and one token (B=1, T=1, C=12)",
            torch.randn(1, 1, N_EMBD),
        )
    )
    results.append(check_variable_sequence_length(model))
    results.append(check_causality(model))
    results.append(check_batch_independence(model))
    results.append(check_uniform_input(model))
    results.append(check_against_reference(model))
    results.append(check_head_divisibility())
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
