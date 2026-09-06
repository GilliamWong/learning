# ARENA validation — 6 September 2026

[Exercise catalog](README.md) · [Your study plan](../LEARNING_PLAN.md)

The import covers every notebook tracked in the pinned ARENA snapshot: **75 notebooks across five chapters**
(34 exercises, 34 references, 7 additional training notebooks). Each was structurally validated, its code
syntax checked, and its local setup plus original import/configuration cell executed in a fresh Jupyter kernel.
**All 75 setup/import checks passed.** This establishes startup, not complete execution of every exercise.

## Complete reference runs

These prepared reference notebooks ran sequentially with the original computation and training settings.
Display pop-outs were suppressed during batch execution; outputs were saved under `validation/sequential`.
Exercise copies were not filled in.

| Reference notebook | Code cells completed | Time | Result |
|---|---:|---:|---|
| 0.4 Backprop | 39 / 39 | 15 s | Passed |
| 1.1 Transformer from Scratch | 55 / 55 | 547 s | Passed, including training and sampling |
| 1.2 Intro to Mech Interp | 50 / 50 | 65 s | Passed |
| 1.4.1 Indirect Object Identification | 70 / 70 | 84 s | Passed |

These runs verify that the provided reference computations execute in this environment. They do not assess
your understanding or establish a new research result.

## Additional runtime checks

- GPU forward and backward calculations passed on the RTX 5070 Ti (16 GB), using PyTorch 2.8.0 + CUDA 12.8.
- ARENA's full transformer loaded GPT-2 weights and matched TransformerLens logits (maximum absolute
  difference 0.000191 in the recorded comparison).
- The induction model loaded and produced causal attention caches for repeated-token inputs.
- IOI caching, an identity intervention, and a finite 12 × 15 residual-patching matrix passed. This extra check
  used one prompt pair; it is a numerical/tooling check, not a paper replication.
- The main ARENA environment's 256 packages and the SAE circuits environment's 149 packages passed
  `uv pip check`. The latter uses a separate kernel because Circuit Tracer requires newer dependency
  versions than the older SAE visualization exercises.
- Circuit Tracer's `ReplacementModel` and `attribute` imports passed; a tiny TransformerLens model completed
  a GPU forward/backward calculation in that kernel. Full gated-model attribution remains manual.
- Petri was pinned to the compatible v2.0.0 source tag. The package and the judge/auditor imports used by
  notebook 4.5 passed. No live investigation was run.
- The ARENA kernel, GPU/gradient output, Plotly heatmap, and interactive CircuitsVis attention view were
  verified in the actual JupyterLab browser interface. Plotly's standard renderer was also installed in
  the Jupyter server environment to fix blank charts. See [Plotly's installation guidance](https://plotly.com/python/getting-started/).
- Your seven saved code implementations in local Modules 1–2 passed the existing workbench checks.
  Their written reflections still need separate review.

Available numerical tests also passed in the following sections. Counts are test invocations, including
different cases of a function. Missing model/data fixtures and deferred extensions are recorded separately.

| Section | Passed checks | Scope remaining |
|---|---:|---|
| Prerequisites | 5 | Full reference run not checked |
| Ray tracing | 6 | Full reference run not checked |
| CNNs / ResNets | 33 | Full training not checked |
| Optimization | 7 | Extra scheduler fixtures; distributed exercises deferred |
| VAEs / GANs | 12 | Dataset downloads and full training not checked |
| Transformer from Scratch | 9 | Full notebook also passed, as above |
| Intro to Mech Interp | 1 | Full notebook also passed, as above |
| Grokking | 7 | Checkpoint/model fixtures and training |
| OthelloGPT | 1 | Model and data fixtures |
| Toy superposition / SAEs | 12 | Full training and model fixtures |
| Intro RL | 5 | Full sequential notebook |
| DQN | 3 | Full training |
| VPG | 9 | Full training |
| PPO | 8 | Full training; optional EnvPool/JAX sections deferred |
| RLHF | 10 | Model fixtures and full training |
| MCTS / AlphaZero | 32 | Full training; individual self-play/training steps passed |
| Inspect | 1 | Live model evaluations |
| LLM agents | 3 | Live model/agent calls |
| AI control | 11 | Live agents and Docker sandbox runs |

## Manual and deferred work

- **Paid APIs:** no live model API calls were made. Validation blocked model API endpoints and external
  writes. Reading API notebooks, implementing local helpers, and inspecting supplied data remain available.
- **Large/gated models:** setup imports passed, but access approvals, large checkpoints, SAE artifacts,
  model-dependent fixtures, and later experiments have not all been exercised. Some defaults exceed this
  machine's 16 GB GPU capacity.
- **Linux/multiple GPUs:** EnvPool, CUDA JAX/Brax, NCCL, and multi-GPU extensions are deferred by your choice.
  Later vLLM-based model workflows also require Linux. WSL was not configured for this course.
- **Long training:** beyond the complete reference runs above, full CNN/GAN/RL/SAE training and the seven
  additional monthly-problem training notebooks have not been run to completion.
- **Agent isolation:** the provided AI-control local test passed with fixed trusted test code. The course's
  live agent sandbox still expects Docker to be configured; model-generated code was not executed locally.
- **Supporting repositories:** ten public repositories were fetched at recorded revisions. Large Git LFS
  artifacts were deferred and may need downloading for later sections.

## Repairs and diagnostic interpretation

Colab installers were replaced by a local setup cell and locked dependencies. Windows DataLoader defaults
use zero worker subprocesses. Malformed placeholder syntax and indentation were repaired without supplying
exercise implementations. The import manifest lists each adaptation.

Three small upstream support repairs are recorded in `support-patches.json`: MCTS sampled-action shape,
Windows paths/interpreter for the explicit local AI-control test branch, and the required world-size argument
in an optional distributed test call. The final distributed call was syntax checked but not run.

The initial helper-import diagnostic is not a substitute for notebook execution: some notebooks initialize
state in earlier cells. For example, Backprop's initial helper probe reported missing registrations, while
its complete sequential run passed all 39 cells. Dataset-generation and investigator-agent helper modules
also refer to earlier notebook prompt variables. Their helper-import reports do not certify or disprove
the full manual notebook workflow. The RLHF probe was rerun using the notebook's import order; its ten
available numerical checks passed. The MCTS probe was rerun with its required search fixture; all 32 passed.

## Evidence and rerunning

- [Import manifest](manifest.json): upstream revision, original hashes, imported paths, and adaptations.
- [Setup/import results](validation/setup-results.json): per-notebook results and exact cell coverage.
- [GPU/model checks](validation/model-checks.json): numerical outputs for the model checks.
- [Circuit Tracer check](validation/circuits-check.json) and [Petri check](validation/petri-check.json): advanced dependency preparation.
- [Browser checks](validation/ui-checks.json): observed kernel, plots, and navigation behavior.
- [Your opening exercise checks](validation/completed-module-checks.json): seven saved implementations.
- `validation/sequential/*.json`: complete-run status, cell counts, timing, and errors if any.
- `validation/reference/*.json`: numerical probes, passed tests, missing fixtures, and deferred tests.

From the workspace root, run `.venv\Scripts\python.exe tools\validate_arena.py` for setup/import checks.
For a full reference notebook, run `.venv\Scripts\python.exe tools\run_arena_reference_notebook.py 1.2_`.
This runs a reference copy and stores outputs separately. Your learner notebook is left untouched.

## Windows launcher

Application Control blocked the previous 45 KB uv virtual-environment launcher. The environment was upgraded
in place using CPython's standard venv launcher, preserving its installed packages; startup now succeeds
through normal Windows PowerShell. The original executables/configuration were backed up in
`.runtime/launcher-backup-20260905_234904` at the workspace root. Windows protection settings were not changed.
