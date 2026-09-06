"""Register ARENA only in this workbench's Jupyter data directory."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for name, directory, label in [("arena", "arena", "ARENA (local GPU)"),
                               ("arena-circuits", "arena/circuits", "ARENA (SAE circuits)")]:
    target = ROOT / ".runtime" / "jupyter" / "data" / "kernels" / name
    target.mkdir(parents=True, exist_ok=True)
    python = ROOT / directory / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        raise FileNotFoundError("Run Setup ARENA.cmd first.")
    spec = {"argv": [str(python), "-X", "utf8", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": label, "language": "python",
            "env": {"PYTHONUTF8": "1", "WANDB_MODE": "disabled", "HF_HUB_DISABLE_TELEMETRY": "1",
                    "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
                    "HF_HOME": str(ROOT / "arena" / ".cache" / "huggingface")}}
    (target / "kernel.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"Registered {label} for this workbench.")
