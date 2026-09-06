"""Shared local setup for the imported notebooks. No installation or API calls here."""

import os
from pathlib import Path
import sys

ARENA_ROOT = Path(__file__).resolve().parent
ARENA_SOURCE = ARENA_ROOT / "ARENA_3.0"


def setup(chapter, section="", notebook=""):
    if chapter not in {f"chapter{i}_{name}" for i, name in enumerate([
        "fundamentals", "transformer_interp", "rl", "llm_evals", "alignment_science"
    ])}:
        raise ValueError(f"Unknown chapter: {chapter}")
    exercises = ARENA_SOURCE / chapter / "exercises"
    section_dir = exercises / section if section else exercises
    if not exercises.is_dir():
        raise FileNotFoundError(f"Missing ARENA support files: {exercises}")
    # Preserve chapter-level utils precedence over same-named section helpers.
    for path in reversed((ARENA_ROOT.parent, exercises, section_dir)):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # Local packages in these sections are intentionally imported from the snapshot.
    for local_package_parent in section_dir.rglob("pyproject.toml") if section_dir.is_dir() else []:
        if local_package_parent.parent.name in {"petri", "assistant-axis", "circuit-tracer"}:
            for candidate in (local_package_parent.parent / "src", local_package_parent.parent):
                if candidate.is_dir() and str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
    os.chdir(section_dir if "monthly_algorithmic_problems" in section else exercises)
    settings = {
        "HF_HOME": ARENA_ROOT / ".cache" / "huggingface",
        "HF_DATASETS_CACHE": ARENA_ROOT / ".cache" / "datasets",
        "MPLCONFIGDIR": ARENA_ROOT / ".cache" / "matplotlib",
        "WANDB_DIR": ARENA_ROOT / "outputs" / "wandb",
    }
    for key, path in settings.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYTHONUTF8", "1")
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    import plotly.io as pio
    pio.renderers.default = "plotly_mimetype"
    from dotenv import load_dotenv
    load_dotenv(ARENA_ROOT / ".env", override=False)
    print(f"ARENA local setup: {chapter} / {section}")
    print(f"Python {sys.version.split()[0]} | PyTorch {torch.__version__} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("Exercise blanks are intentional. Paid API sections are manual; see arena/README.md.")
    return {"ARENA_ROOT": ARENA_ROOT, "ARENA_SOURCE": ARENA_SOURCE, "IN_COLAB": False,
            "repo": ARENA_SOURCE.name, "branch": "527f9376b40ad9a12ecd80490884b0009b54dd55",
            "chapter": chapter, "root": str(ARENA_SOURCE), "root_dir": ARENA_SOURCE,
            "exercises_dir": exercises, "section_dir": section_dir}
