"""Contract checks for the opening exercises. These never mark progress complete."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _tensor_cases(task):
    if task == "batches":
        return [
            ((torch.arange(12), 4), torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])),
            ((torch.arange(16)[::2], 2), torch.tensor([[0, 2], [4, 6], [8, 10], [12, 14]])),
        ]
    if task == "center":
        return [
            ((torch.tensor([[2., 4.], [8., 12.], [1., 1.]]),), torch.tensor([[-1., 1.], [-2., 2.], [0., 0.]])),
            ((torch.tensor([[1., 2., 6.], [-4., -1., 2.]]),), torch.tensor([[-2., -1., 3.], [-3., 0., 3.]])),
        ]
    if task == "similarity":
        q = 1 / math.sqrt(2)
        return [
            ((torch.tensor([[1., 0.], [0., 2.], [1., 1.], [0., 0.]]),),
             torch.tensor([[1., 0., q, 0.], [0., 1., q, 0.], [q, q, 1., 0.], [0., 0., 0., 0.]])),
            ((torch.tensor([[1., -1., 0.], [-1., 1., 0.]]),), torch.tensor([[1., -1.], [-1., 1.]])),
        ]
    if task == "probabilities":
        return [
            ((torch.tensor([[1000., 1001.], [0., 0.]]),), torch.tensor([[0.26894142, 0.73105858], [0.5, 0.5]])),
            ((torch.tensor([[[0., 0., 0.], [1., 1., 1.]], [[-1000., -1000., -1000.], [8., 8., 8.]]]),), torch.full((2, 2, 3), 1 / 3)),
        ]
    raise KeyError(task)


def _tensors(task, function):
    for index, (args, expected) in enumerate(_tensor_cases(task), 1):
        snapshots = [arg.clone() for arg in args if isinstance(arg, torch.Tensor)]
        print(f"Case {index}")
        for arg in args:
            print(f"  input {tuple(arg.shape)}:\n{arg}" if isinstance(arg, torch.Tensor) else f"  batch size: {arg}")
        actual = function(*args)
        if actual is None:
            print("Not implemented yet. Replace the function's return None, then rerun its definition and this check.")
            return False
        assert isinstance(actual, torch.Tensor), "Return a torch.Tensor."
        print(f"  expected {tuple(expected.shape)}:\n{expected}")
        print(f"  received {tuple(actual.shape)}:\n{actual}")
        assert actual.shape == expected.shape, f"Expected shape {tuple(expected.shape)}, received {tuple(actual.shape)}."
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for before, after in zip(snapshots, [arg for arg in args if isinstance(arg, torch.Tensor)]):
            torch.testing.assert_close(after, before, msg="The input changed in place. Return a result while preserving the input.")
    if task == "probabilities":
        x = torch.tensor([[0.2, -0.4, 0.8]], requires_grad=True)
        y = function(x)
        assert y.requires_grad, "The result must remain connected to autograd."
        grad, = torch.autograd.grad(y[0, 0], x)
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0, "Check that gradients are finite and nonzero."
        print("Autograd connection: PASS")
    return True


def _model(function):
    model = function()
    if model is None:
        print("Not implemented yet. Return a trainable nn.Module.")
        return False
    assert isinstance(model, nn.Module), "Return an nn.Module."
    for batch in (1, 7):
        output = model(torch.randn(batch, 2))
        assert output.shape == (batch, 2), f"Expected ({batch}, 2) class scores, got {tuple(output.shape)}."
        assert torch.isfinite(output).all(), "Class scores must be finite."
        assert output.requires_grad, "Outputs must be connected to trainable parameters."
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0, "The model needs trainable parameters."
    print("Batch sizes 1 and 7: correct output shape, finite values, and trainable parameters.")
    return True


def _step(function):
    torch.manual_seed(17)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    x = torch.tensor([[1., 0.], [0., 1.], [-1., -1.]])
    y = torch.tensor([0, 1, 1])
    params = list(model.parameters())
    for index in range(2):
        model.eval()
        for p in params:
            p.grad = torch.ones_like(p)  # a previous batch must not leak into this step
        expected_loss = F.cross_entropy(model(x), y)
        expected_grads = torch.autograd.grad(expected_loss, params)
        before = [p.detach().clone() for p in params]
        loss = function(model, optimizer, x, y)
        if loss is None:
            print("Not implemented yet. Implement one step and return its loss as a Python number.")
            return False
        assert isinstance(loss, (float, int)), "Return a Python number, not a tensor attached to a computation graph."
        assert math.isclose(loss, expected_loss.item(), rel_tol=1e-5), "Return cross-entropy loss for this batch before its parameter update."
        assert model.training, "A training step should put the model in training mode."
        for old, p, grad in zip(before, params, expected_grads):
            torch.testing.assert_close(p, old - 0.03 * grad, rtol=1e-5, atol=1e-6,
                                       msg="The parameter update differs from this batch's gradient. Check gradient clearing and optimizer use.")
        print(f"Step {index + 1}: loss={loss:.6f}; correct parameter update with no stale gradient accumulation.")
    return True


def _evaluate(function):
    class Probe(nn.Module):
        def forward(self, x):
            self.seen_training = self.training
            self.seen_grad = torch.is_grad_enabled()
            return x
    model = Probe()
    scores = torch.tensor([[4., -1.], [0., 2.], [3., 1.]])
    labels = torch.tensor([0, 1, 1])
    result = function(model, scores, labels)
    if result is None:
        print("Not implemented yet. Return a dictionary with loss and accuracy.")
        return False
    assert isinstance(result, dict), "Return {'loss': number, 'accuracy': number}."
    expected_loss = F.cross_entropy(scores, labels).item()
    assert isinstance(result.get("loss"), (float, int)), "Loss should be a Python number."
    assert isinstance(result.get("accuracy"), (float, int)), "Accuracy should be a Python number."
    assert math.isclose(result["loss"], expected_loss, rel_tol=1e-5), "Use cross entropy on the raw class scores."
    assert math.isclose(result["accuracy"], 2 / 3, rel_tol=1e-5), "Accuracy is the fraction of correctly classified examples."
    assert not model.seen_training, "Evaluate in evaluation mode."
    assert not model.seen_grad, "Evaluation should not build an autograd graph."
    print(f"Expected / received loss: {expected_loss:.6f} / {result['loss']:.6f}")
    print(f"Expected / received accuracy: {2 / 3:.6f} / {result['accuracy']:.6f}")
    print("Evaluation mode and disabled gradient recording: PASS")
    return True


def check(task, function):
    """Print useful diagnostics. False means unfinished or incorrect, not a notebook crash."""
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            if task in {"batches", "center", "similarity", "probabilities"}:
                passed = _tensors(task, function)
            else:
                passed = {"model": _model, "step": _step, "evaluate": _evaluate}[task](function)
        if passed:
            print("PASS — now explain why it works before checking off the task.")
        return bool(passed)
    except NotImplementedError:
        print("Not implemented yet. Work on the function above, then rerun this check.")
    except Exception as error:
        print(f"NOT YET — {type(error).__name__}: {error}")
    return False
