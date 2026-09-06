"""Build the plain Markdown ARENA indexes from the pinned import and check reports."""

import json
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
ARENA = ROOT / "arena"
CHAPTERS = {
    "chapter0_fundamentals": "0 · Fundamentals",
    "chapter1_transformer_interp": "1 · Transformer interpretability",
    "chapter2_rl": "2 · Reinforcement learning",
    "chapter3_llm_evals": "3 · LLM evaluations",
    "chapter4_alignment_science": "4 · Alignment science",
}
NOTES = {
    "0.3": "Core optimizer tests passed. Distributed/NCCL extras deferred.",
    "0.5": "12 component tests passed; dataset downloads and full training remain to run.",
    "1.1": "Start here. Full reference run passed, including training and sampling.",
    "1.2": "Full reference run passed. Follow after 1.1.",
    "1.3.1": "Later branch. Gated models and substantial memory requirements.",
    "1.3.2": "Later branch. Model downloads and optional remote/API execution.",
    "1.3.3": "Later branch. Pretrained SAEs, model access, and optional paid APIs.",
    "1.3.4": "Later branch. Oracle checkpoints and larger model requirements.",
    "1.4.1": "Full reference run passed. Follow induction and a small extension.",
    "1.4.2": "Separate SAE circuits kernel prepared and checked. Gated model/SAE artifacts; full execution remains manual.",
    "1.5.1": "Later small-model circuit investigation; full reference run not yet checked.",
    "1.5.2": "Later branch. Numerical helpers checked; checkpoints/training remain to run.",
    "1.5.3": "Later branch. Model/data downloads; full reference run not yet checked.",
    "1.5.4": "Recommended representations branch. 12 numerical checks passed; full training not yet checked.",
    "2.3": "Core PPO numerical checks passed. EnvPool and JAX/Brax extensions deferred.",
    "2.4": "10 numerical checks passed. Full RLHF training and model fixtures not yet checked.",
    "2.5": "32 component checks passed, including self-play/training steps. Full AlphaZero training not run.",
    "3.1": "Paid model calls are manual.",
    "3.2": "Paid data-generation and evaluation calls are manual.",
    "3.3": "Inspect setup checked; live evaluations are manual.",
    "3.4": "Local helper checks passed; model/agent calls are manual.",
    "3.5": "11 local checks passed. Live agents and Docker sandbox experiments remain manual.",
    "4.1": "Manual API/model access; training and evaluation not run.",
    "4.2": "Manual API/model access; large model experiments not run.",
    "4.3": "Manual API/model access; reasoning-model experiments not run.",
    "4.4": "Manual API/model access; large model experiments not run.",
    "4.5": "Compatible Petri imports checked. Live API investigations remain manual.",
}


def link(record, base=ARENA):
    path = (ROOT / record["local"]).relative_to(base).as_posix()
    label = Path(path).stem.removesuffix("_exercises").removesuffix("_solutions").replace("_", " ")
    return f"[{label}]({quote(path, safe='/')})"


def main():
    manifest = json.loads((ARENA / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["notebooks"]
    setup_path = ARENA / "validation/setup-results.json"
    setup = json.loads(setup_path.read_text(encoding="utf-8"))["results"] if setup_path.exists() else []
    passed = sum(r["status"] == "passed" for r in setup)
    intro = f"""# ARENA notebooks

**Your next notebook is [1.1 · Transformer from Scratch](notebooks/chapter1_transformer_interp/part1_transformer_from_scratch/1.1_Transformer_from_Scratch_exercises.ipynb).**
Use [your study plan](../LEARNING_PLAN.md) for the reading, coding, and experiment order.

## Open and work

1. Open an exercise notebook below. The appropriate ARENA kernel is selected automatically.
2. Run the local setup cell first, then work through the notebook with **Shift+Enter**.
3. Implement each exercise before running its tests. Blank answers and `NotImplementedError` are intentional.
4. Save your code and writing with **Ctrl+S**. Use the workbench module checklists for your milestones.

The environment is installed. [Check the GPU and plots](00_Environment_Check.ipynb) if something looks wrong.
Most notebooks use **ARENA (local GPU)**. Notebook 1.4.2 uses **ARENA (SAE circuits)** because its package
versions differ from the older exercises. You do not need to switch versions manually.
Use **Setup ARENA.cmd** in the workspace folder only if you need to rebuild it. Restart a kernel when switching
between chapters, because different chapters have helpers with the same names. Close unused kernels to free GPU memory.

## What has been checked

Imported **75 notebooks: 34 exercises, 34 references, and 7 additional training notebooks**, covering all five chapters.
**{passed}/75 setup and import checks passed.** Backprop, Transformer from Scratch, Intro to Mech Interp, and IOI
reference notebooks also passed complete sequential runs. More numerical checks are recorded in [validation details](VALIDATION.md).

Full execution has not been verified for every advanced notebook. Large models, gated checkpoints, long training runs,
and paid API sections have extra requirements. Paid calls stay manual, and Linux-only/multi-GPU extensions are deferred
by your choice. The notes below identify these boundaries before you start a branch.

References are in a [separate solutions index](reference/README.md), and the seven extra notebooks have an
[additional training index](additional/README.md).
"""
    lines = [intro]
    for chapter, title in CHAPTERS.items():
        lines += [f"\n## {title}\n", "| Exercise notebook | Current scope |", "|---|---|"]
        for r in records:
            if r["kind"] == "exercise" and r["chapter"] == chapter:
                number = Path(r["local"]).name.split("_")[0]
                note = NOTES.get(number, "Local numerical checks passed; full reference run not yet checked.")
                if number == "0.4":
                    note = "Full sequential reference run passed."
                lines.append(f"| {link(r)} | {note} |")
    lines += ["""
## APIs and advanced sections

No paid calls were made during setup or validation. When you choose to run an API section, copy `.env.example`
to `.env` in this `arena` folder and fill only the keys that section needs. Keep that file private; it is ignored
by Git. Restart the notebook kernel and rerun setup. Choose model, request count, and spending limits before
starting a live evaluation. A notebook cell can make many requests, so work through API notebooks cell by cell.

This Windows environment includes core RL, Gymnasium, and MuJoCo. The course's optional EnvPool and CUDA
JAX/Brax sections need a later Linux environment; NCCL/distributed GPU exercises also need appropriate hardware.
Some later high-throughput model tools, including vLLM-based assistant-axis workflows, also need Linux.
Docker-based agent sandboxes need Docker configured before use. The fixed local sandbox test was checked with
trusted test code; running model-generated code belongs in the intended isolated sandbox.

Supporting repositories have been fetched, but large Git LFS artifacts are not downloaded automatically.
Gated Hugging Face models need your own access approval. The installed GPU has 16 GB VRAM, so some larger-model
sections will need a different model configuration or more compute. Those sections are prepared, not certified
to fit this machine unchanged.

## Source and local changes

""", f"Source: [official ARENA snapshot `{manifest['revision'][:12]}`]({manifest['upstream']}/tree/{manifest['revision']}).\n",
              "Original notebooks remain in `ARENA_3.0`. Prepared copies replace Colab installers and paths, use Windows-compatible DataLoader defaults, and repair malformed placeholders. Exercise answers remain blank.\n",
              "The import manifest records source links, hashes, and adaptations. Small support-code repairs are recorded in `support-patches.json`; fetched dependencies in `supporting-repositories.json`. Source content and attribution remain with the snapshot.\n"]
    (ARENA / "README.md").write_text("\n".join(lines), encoding="utf-8")
    for kind, folder, title in [("solution", "reference", "Reference solutions"), ("additional", "additional", "Additional training notebooks")]:
        lines = [f"# {title}\n", "[Back to exercise catalog](../README.md)\n"]
        if kind == "solution":
            lines += ["These contain completed answers. Try the exercises and hints first; open a reference when you choose to compare. Validation outputs are stored separately from these copies.\n"]
        else:
            lines += ["These are the seven monthly-problem training notebooks tracked by the imported ARENA snapshot. Setup/import checks passed; their full training runs have not been verified.\n"]
        for r in records:
            if r["kind"] == kind:
                context = Path(r["source"]).parent.name.replace("_", " ")
                lines.append(f"- {link(r, ARENA / folder)} — {context}")
        (ARENA / folder / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Built ARENA exercise, reference, and additional notebook indexes.")


if __name__ == "__main__":
    main()
