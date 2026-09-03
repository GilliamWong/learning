# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "tensorboard",
#     "torch>=2.2",
# ]
# ///
"""Print and validate the Transformer Block: sublayers, residuals, pre-norm.

Block owns one CausalSelfAttention and one MLP, each wrapped in its own
LayerNorm and residual connection. The cases below check that wrapping, not the
attention maths -- run check_attention.py first, since a broken attention will
fail here too and the message will be less specific.

Run from the repository root with:

    uv run exercises/karpathy/with-headers/makemore/check_block.py
"""

from __future__ import annotations

import sys
import traceback

import torch
import torch.nn as nn

from makemore import Block, CausalSelfAttention, ModelConfig, NewGELU


BLOCK_SIZE = 8
N_EMBD = 12
N_HEAD = 3


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


def check_structure(model: Block) -> bool:
    print_heading("Sublayer inventory")
    print("submodules:")
    for name, module in model.named_modules():
        if name:
            print(f"  {name:<28} {type(module).__name__}")

    modules = [m for name, m in model.named_modules() if name]
    attentions = [m for m in modules if isinstance(m, CausalSelfAttention)]
    layer_norms = [m for m in modules if isinstance(m, nn.LayerNorm)]
    gelus = [m for m in modules if isinstance(m, (NewGELU, nn.GELU))]
    linears = [m for m in modules if isinstance(m, nn.Linear)]
    expanding = [m for m in linears if m.out_features == 4 * N_EMBD]
    contracting = [m for m in linears if m.in_features == 4 * N_EMBD]

    print(f"\nCausalSelfAttention modules: {len(attentions)} (expected 1)")
    print(f"LayerNorm modules:           {len(layer_norms)} (expected 2)")
    print(f"GELU modules:                {len(gelus)} (expected 1)")
    print(f"Linear(C -> 4C) modules:     {len(expanding)} (expected 1)")
    print(f"Linear(4C -> C) modules:     {len(contracting)} (expected 1)")

    if len(gelus) == 0:
        print("no GELU found -- the MLP needs a nonlinearity between its two Linear")
        print("layers, otherwise the whole feed-forward branch collapses to one")
        print("linear map and adds nothing attention cannot already do.")
    if not expanding or not contracting:
        print("the MLP inside a Block expands to 4 * n_embd and comes back down.")
        print("Note it uses 4 * n_embd, not config.n_embd2 -- n_embd2 is for the")
        print("RNN and MLP models and is unused on the transformer path.")

    attention_params = 4 * N_EMBD * N_EMBD + 4 * N_EMBD
    mlp_params = 8 * N_EMBD * N_EMBD + 5 * N_EMBD
    norm_params = 4 * N_EMBD
    expected = attention_params + mlp_params + norm_params
    total = sum(p.numel() for p in model.parameters())
    print(f"\nparameter total: {total}")
    print(f"expected {expected} = attention {attention_params} + mlp {mlp_params}"
          f" + 2 LayerNorm {norm_params}")

    structure_ok = (
        len(attentions) == 1
        and len(layer_norms) == 2
        and len(gelus) == 1
        and len(expanding) == 1
        and len(contracting) == 1
    )
    count_ok = total == expected
    if not count_ok:
        print("(a mismatch here is worth reading alongside the module list above;")
        print("bias=False anywhere will also shift the count)")

    passed = structure_ok and count_ok
    result_line(passed)
    return passed


def check_forward(model: Block, name: str, x: torch.Tensor) -> bool:
    print_heading(name)
    print(f"input shape:  {tuple(x.shape)}")
    print(f"input dtype:  {x.dtype}")

    try:
        actual = model(x)

        if not isinstance(actual, torch.Tensor):
            raise TypeError(f"Block returned {type(actual).__name__}, not torch.Tensor")

        print(f"output shape: {tuple(actual.shape)}")
        print(f"output dtype: {actual.dtype}")
        print(f"output:\n{actual}")

        shape_ok = actual.shape == x.shape
        dtype_ok = actual.dtype == x.dtype
        finite_ok = bool(torch.isfinite(actual).all())

        print(f"\nshape preserved: {shape_ok}")
        print(f"dtype preserved: {dtype_ok}")
        print(f"all values finite: {finite_ok}")

        if not shape_ok:
            print("blocks are stacked n_layer deep, so a block must hand the next one")
            print("exactly the shape it received.")

        passed = shape_ok and dtype_ok and finite_ok
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def check_residual_and_prenorm(model: Block) -> bool:
    print_heading("Residual connections and pre-norm placement")
    print("Feed a large-magnitude input. With pre-norm residuals the block computes")
    print("x + attn(ln_1(x)) + mlp(ln_2(x)): the two branches see normalised input,")
    print("so they contribute O(1) no matter how big x is, and the output stays")
    print("close to x. Two ways this fails:")
    print("  - no residual at all: output is O(1) while the input is O(1000), so the")
    print("    relative difference is about 1")
    print("  - post-norm (LayerNorm outside the residual): the norm rescales the sum")
    print("    back to unit scale, again giving a relative difference near 1")

    try:
        rows = []
        for scale in (1.0, 100.0, 1000.0):
            x = torch.randn(2, BLOCK_SIZE, N_EMBD) * scale
            y = model(x)
            relative = ((y - x).norm() / x.norm()).item()
            rows.append((scale, relative))
            print(f"  input scale {scale:>7.1f}:  ||y - x|| / ||x|| = {relative:.6f}")

        largest = rows[-1][1]
        decreasing = rows[0][1] > rows[1][1] > rows[2][1]
        print(f"\nrelative difference at the largest scale: {largest:.6f} (want < 0.05)")
        print(f"relative difference shrinks as the input grows: {decreasing}")
        if largest >= 0.05:
            print("the output is not tracking the input. Check that both sublayers are")
            print("written as `x = x + sublayer(norm(x))`, with the LayerNorm inside")
            print("the branch rather than applied to the sum.")

        passed = largest < 0.05 and decreasing
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def check_both_branches_used(model: Block):
    print_heading("Both sublayers actually contribute")
    print("Zero out one branch at a time by zeroing the weights and biases of the")
    print("modules that feed it, and confirm the output changes. A branch computed")
    print("but never added to x -- a missing `x = x + ...` -- shows up as no change.")

    x = torch.randn(2, BLOCK_SIZE, N_EMBD)
    try:
        baseline = model(x)
        original = {name: p.detach().clone() for name, p in model.named_parameters()}

        def restore() -> None:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(original[name])

        results = []
        for label, predicate in (
            ("attention", lambda name: "attn" in name.lower()),
            ("mlp", lambda name: "mlp" in name.lower() or "fc" in name.lower()),
        ):
            targets = [name for name, _ in model.named_parameters() if predicate(name)]
            if not targets:
                print(f"  {label}: no parameters matched by name, skipping this half")
                continue
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if predicate(name):
                        parameter.zero_()
            delta = (model(x) - baseline).abs().max().item()
            restore()
            ok = delta > 0.0
            print(f"  zeroing {label} ({len(targets)} tensors) changed output by: "
                  f"{delta:.3e} ({'ok' if ok else 'NO EFFECT'})")
            results.append(ok)

        if not results:
            print("\ncould not match either branch by parameter name; naming the")
            print("submodules attn/mlp (as minGPT does) makes this case meaningful.")
            print("\nCASE RESULT: SKIP")
            return None

        passed = all(results)
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def check_causality(model: Block) -> bool:
    print_heading("Causality survives the whole block")
    print("The MLP is position-wise and the residual is elementwise, so neither can")
    print("move information across positions. The block must be exactly as causal as")
    print("the attention inside it.")

    seq_len = BLOCK_SIZE
    x = torch.randn(2, seq_len, N_EMBD, requires_grad=True)
    try:
        future_ok = []
        for t in range(seq_len):
            if x.grad is not None:
                x.grad = None
            y = model(x)
            y[:, t, :].sum().backward()
            assert x.grad is not None
            future = x.grad[:, t + 1 :, :].abs().max().item() if t + 1 < seq_len else 0.0
            leak = future == 0.0
            future_ok.append(leak)
            print(f"  output position {t}: max |grad| from future = {future:.3e} "
                  f"({'ok' if leak else 'LEAK'})")

        passed = all(future_ok)
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def check_variable_sequence_length(model: Block) -> bool:
    print_heading("Variable sequence length T = 1 .. block_size")
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


def check_gradients(model: Block) -> bool:
    print_heading("Every parameter receives a gradient")
    print("A parameter with no gradient was built in __init__ and never used in")
    print("forward -- a LayerNorm that was skipped, or an MLP layer left out.")

    model.zero_grad(set_to_none=True)
    x = torch.randn(2, BLOCK_SIZE, N_EMBD)
    try:
        model(x).sum().backward()
        results = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                print(f"  {name:<28} NO GRADIENT")
                results.append(False)
                continue
            magnitude = parameter.grad.abs().sum().item()
            ok = magnitude > 0.0
            print(f"  {name:<28} sum |grad| = {magnitude:.6e} {'ok' if ok else 'ZERO'}")
            results.append(ok)
        model.zero_grad(set_to_none=True)

        passed = bool(results) and all(results)
        result_line(passed)
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def check_determinism(model: Block) -> bool:
    print_heading("Deterministic, and identical in train and eval mode")
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

    print_heading("Block diagnostic")
    print(f"Python:  {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"config:  block_size={BLOCK_SIZE}, n_embd={N_EMBD}, n_head={N_HEAD}")
    print("Run check_attention.py first: Block contains CausalSelfAttention, so a")
    print("failure there will also fail cases here.")

    config = make_config()
    try:
        model = Block(config)
    except Exception as error:
        print_heading("Construction")
        print("Block(config) failed, so no other case can run.")
        print_exception(error)
        print("\nOVERALL RESULT: FAIL")
        return 1

    results = []
    results.append(check_structure(model))
    results.append(
        check_forward(
            model,
            "Forward with shape (B=2, T=block_size=8, C=12)",
            torch.randn(2, BLOCK_SIZE, N_EMBD),
        )
    )
    results.append(check_variable_sequence_length(model))
    results.append(check_residual_and_prenorm(model))
    results.append(check_both_branches_used(model))
    results.append(check_causality(model))
    results.append(check_gradients(model))
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
