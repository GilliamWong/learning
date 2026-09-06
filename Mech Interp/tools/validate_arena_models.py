"""Check the actual GPU model path used by the next mech-interp lessons."""

import gc
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
runtime = runpy.run_path(str(ROOT / "arena" / "runtime.py"))
runtime["setup"]("chapter1_transformer_interp", "part1_transformer_from_scratch")
from validate_arena import READ_ONLY_NETWORK
exec(READ_ONLY_NETWORK, {})

import torch
from transformer_lens import HookedTransformer, patching
from part1_transformer_from_scratch.solutions import Config, DemoTransformer

results = []
torch.manual_seed(12)
assert torch.cuda.is_available()
reference = HookedTransformer.from_pretrained("gpt2-small", device="cuda", fold_ln=False,
                                             center_unembed=False, center_writing_weights=False)
reference.eval()
tokens = reference.to_tokens("A small model can still teach us a great deal.")
student_architecture = DemoTransformer(Config(debug=False)).to("cuda").eval()
student_architecture.load_state_dict(reference.state_dict(), strict=False)
with torch.inference_mode():
    expected = reference(tokens)
    actual = student_architecture(tokens)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    results.append({"check": "ARENA 1.1 full transformer matches pretrained GPT-2 logits", "passed": True,
                    "shape": list(actual.shape), "max_absolute_error": (actual - expected).abs().max().item()})
print("GPT-2 architecture and pretrained-weight comparison passed.", flush=True)
del student_architecture, actual, expected
gc.collect()
torch.cuda.empty_cache()

clean = reference.to_tokens("When John and Mary went to the store, John gave a drink to")
corrupt = reference.to_tokens("When John and Mary went to the store, Mary gave a drink to")
assert clean.shape == corrupt.shape
mary = reference.to_single_token(" Mary")
john = reference.to_single_token(" John")
def gap(logits):
    return (logits[:, -1, mary] - logits[:, -1, john]).mean()
with torch.inference_mode():
    clean_logits, cache = reference.run_with_cache(clean)
    corrupt_logits = reference(corrupt)
    restored = reference.run_with_hooks(clean, fwd_hooks=[("blocks.5.hook_resid_pre", lambda act, hook: cache["blocks.5.hook_resid_pre"])])
    torch.testing.assert_close(restored, clean_logits, atol=1e-5, rtol=1e-5)
    patches = patching.get_act_patch_resid_pre(reference, corrupt, cache, gap)
    assert patches.shape == (reference.cfg.n_layers, clean.shape[1])
    assert torch.isfinite(patches).all()
    results.append({"check": "IOI cache, identity intervention, and residual-stream activation patching", "passed": True,
                    "patch_matrix_shape": list(patches.shape), "clean_logit_gap": gap(clean_logits).item(),
                    "corrupt_logit_gap": gap(corrupt_logits).item(), "scope": "One-prompt API/numerical check, not a full IOI replication."})
print("IOI caching and the full layer/position patch matrix passed.", flush=True)
del reference, clean_logits, corrupt_logits, restored, cache, patches
gc.collect()
torch.cuda.empty_cache()

induction = HookedTransformer.from_pretrained("attn-only-2l", device="cuda")
induction.eval()
sequence = torch.randint(100, 10000, (2, 32), device="cuda")
bos = torch.full((2, 1), induction.tokenizer.bos_token_id, dtype=torch.long, device="cuda")
repeated = torch.cat([bos, sequence, sequence], dim=1)
with torch.inference_mode():
    logits, cache = induction.run_with_cache(repeated)
    losses = induction(repeated, return_type="loss", loss_per_token=True)
    pattern = cache["pattern", 1]
    assert logits.shape[:2] == repeated.shape and losses.shape == (2, 64)
    assert torch.isfinite(logits).all() and torch.isfinite(losses).all()
    assert torch.triu(pattern, diagonal=1).abs().max() == 0
    results.append({"check": "ARENA 1.2 attention-only model and repeated-token cache", "passed": True,
                    "logits_shape": list(logits.shape), "attention_shape": list(pattern.shape),
                    "first_half_loss": losses[:, :32].mean().item(), "second_half_loss": losses[:, 32:].mean().item()})
print("Induction model, causal attention, and repeated-token loss passed.", flush=True)
(ROOT / "arena" / "validation" / "model-checks.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2), flush=True)
