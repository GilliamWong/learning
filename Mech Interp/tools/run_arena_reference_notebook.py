"""Execute a complete reference notebook, preserving sequential notebook state."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient

from launch import configure_environment
from validate_arena import READ_ONLY_NETWORK

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="Notebook filename prefix, e.g. 0.4_")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    configure_environment()
    path = next((ROOT / "arena" / "reference").glob(f"**/{args.prefix}*solutions.ipynb"))
    notebook = nbformat.read(path, as_version=4)
    quiet_plots = '''
import plotly.basedatatypes
plotly.basedatatypes.BaseFigure.show = lambda self, *args, **kwargs: None
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None
from PIL import Image
Image.Image.show = lambda self, *args, **kwargs: None
'''
    notebook.cells.insert(0, nbformat.v4.new_code_cell(READ_ONLY_NETWORK + quiet_plots))
    started = time.monotonic()
    result = {"notebook": path.relative_to(ROOT).as_posix(), "coverage": "sequential reference notebook; only display pop-outs suppressed",
              "status": "passed", "error": None}
    try:
        NotebookClient(notebook, timeout=args.timeout, kernel_name=notebook.metadata.kernelspec.name,
                       resources={"metadata": {"path": str(path.parent)}}).execute()
    except Exception as error:
        result["status"] = "manual_credentials" if any(k in str(error) for k in ["API_KEY", "API key", "MANUAL_API", "MANUAL_NETWORK_WRITE", "HF_TOKEN", "gated repo"]) else "failed"
        result["error"] = str(error)[-7000:]
    result["seconds"] = round(time.monotonic() - started, 2)
    result["executed_code_cells"] = sum(c.cell_type == "code" and c.execution_count is not None for c in notebook.cells) - 1
    result["total_code_cells"] = sum(c.cell_type == "code" for c in notebook.cells) - 1
    result["checked_at"] = datetime.now(timezone.utc).isoformat()
    out = ROOT / "arena" / "validation" / "sequential"
    out.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, out / path.name)
    (out / (path.stem + ".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
