# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "tensorboard",
#     "torch>=2.2",
# ]
# ///
"""Print and validate NewGELU inputs, outputs, errors, and gradients.

Run from the repository root with:

    uv run exercises/karpathy/with-headers/makemore/check_gelu.py
"""

from __future__ import annotations

import sys
import traceback

import torch
from torch.nn import functional as F

from makemore import NewGELU


ATOL = 1e-6
RTOL = 1e-5


def print_heading(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def print_exception(error: BaseException) -> None:
    print(f"ERROR TYPE: {type(error).__name__}")
    print(f"ERROR MESSAGE: {error}")
    print("TRACEBACK:")
    traceback.print_exc(file=sys.stdout)


def check_forward(model: NewGELU, name: str, x: torch.Tensor) -> bool:
    print_heading(name)
    print(f"input shape:  {tuple(x.shape)}")
    print(f"input dtype:  {x.dtype}")
    print(f"input device: {x.device}")
    print(f"input:\n{x}")

    expected = F.gelu(x, approximate="tanh")
    print(f"\nexpected output from PyTorch tanh-GELU:\n{expected}")

    try:
        actual = model(x)
        print(f"\nactual output from NewGELU:\n{actual}")

        if not isinstance(actual, torch.Tensor):
            raise TypeError(
                f"NewGELU returned {type(actual).__name__}, not torch.Tensor"
            )

        print(f"\nactual shape:  {tuple(actual.shape)}")
        print(f"actual dtype:  {actual.dtype}")
        print(f"actual device: {actual.device}")

        shape_ok = actual.shape == x.shape
        dtype_ok = actual.dtype == x.dtype
        device_ok = actual.device == x.device
        finite_ok = bool(torch.isfinite(actual).all())

        print(f"shape preserved: {shape_ok}")
        print(f"dtype preserved: {dtype_ok}")
        print(f"device preserved: {device_ok}")
        print(f"all values finite: {finite_ok}")

        if not shape_ok:
            raise AssertionError(
                f"shape changed from {tuple(x.shape)} to {tuple(actual.shape)}"
            )

        absolute_error = (actual - expected).abs()
        print(f"\nabsolute error at every element:\n{absolute_error}")
        print(f"maximum absolute error: {absolute_error.max().item():.9g}")

        try:
            torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
            values_ok = True
            print(f"value comparison: PASS (atol={ATOL}, rtol={RTOL})")
        except AssertionError as error:
            values_ok = False
            print(f"value comparison: FAIL (atol={ATOL}, rtol={RTOL})")
            print(error)

        passed = shape_ok and dtype_ok and device_ok and finite_ok and values_ok
        print(f"\nCASE RESULT: {'PASS' if passed else 'FAIL'}")
        return passed
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def check_gradients(model: NewGELU) -> bool:
    print_heading("Gradient check with shape (B=1, T=2, C=4)")
    values = torch.tensor(
        [[[-3.0, -1.0, 0.0, 1.0], [2.0, -2.0, 0.5, -0.5]]],
        dtype=torch.float64,
    )
    actual_input = values.clone().requires_grad_(True)
    reference_input = values.clone().requires_grad_(True)

    reference_output = F.gelu(reference_input, approximate="tanh")
    reference_output.sum().backward()

    print(f"input:\n{values}")
    print(f"\nexpected gradient:\n{reference_input.grad}")

    try:
        actual_output = model(actual_input)
        actual_output.sum().backward()
        actual_gradient = actual_input.grad
        print(f"\nactual gradient:\n{actual_gradient}")

        if actual_gradient is None:
            raise AssertionError("NewGELU did not produce a gradient for its input")

        gradient_error = (actual_gradient - reference_input.grad).abs()
        print(f"\nabsolute gradient error:\n{gradient_error}")
        print(f"maximum gradient error: {gradient_error.max().item():.9g}")

        try:
            torch.testing.assert_close(
                actual_gradient,
                reference_input.grad,
                atol=ATOL,
                rtol=RTOL,
            )
            print(f"gradient comparison: PASS (atol={ATOL}, rtol={RTOL})")
            print("\nCASE RESULT: PASS")
            return True
        except AssertionError as error:
            print(f"gradient comparison: FAIL (atol={ATOL}, rtol={RTOL})")
            print(error)
            print("\nCASE RESULT: FAIL")
            return False
    except Exception as error:
        print("\nCASE RESULT: ERROR")
        print_exception(error)
        return False


def main() -> int:
    torch.set_printoptions(precision=6, linewidth=160, threshold=10_000)
    torch.manual_seed(3407)

    print_heading("NewGELU diagnostic")
    print(f"Python:  {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Reference: torch.nn.functional.gelu(..., approximate='tanh')")

    model = NewGELU()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"NewGELU parameter count: {parameter_count} (expected: 0)")

    cases = [
        (
            "Simple values with shape (N=7,)",
            torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]),
        ),
        (
            "Residual-stream example with shape (B=2, T=3, C=4)",
            torch.linspace(-3.0, 3.0, steps=2 * 3 * 4).reshape(2, 3, 4),
        ),
        (
            "Expanded-MLP example with shape (B=2, T=3, 4C=16)",
            torch.linspace(-4.0, 4.0, steps=2 * 3 * 16).reshape(2, 3, 16),
        ),
    ]

    results = [check_forward(model, name, x) for name, x in cases]
    results.append(check_gradients(model))

    print_heading("Summary")
    passed = sum(results)
    total = len(results)
    print(f"passed: {passed}/{total}")
    if all(results) and parameter_count == 0:
        print("OVERALL RESULT: PASS")
        return 0

    print("OVERALL RESULT: FAIL")
    print("The command exits with status 1 so terminal tooling can detect the failure.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
