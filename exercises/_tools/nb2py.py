#!/usr/bin/env python3
"""
nb2py.py — convert a Jupyter notebook (.ipynb) to a standard Python file.

Two modes:
  * solution mode (default): emit every code cell verbatim (IPython magics and
    shell escapes are commented out so the file is valid Python). Markdown cells
    become `# ` comment blocks. Cells are delimited with `# %%` markers so the
    result still runs cell-by-cell in VS Code / Jupyter.
  * --strip: produce an exercise skeleton. Markdown (the lecture narration) and
    import/comment lines are kept; every other code line is replaced by a single
    `# TODO` so you reconstruct the cell yourself.

Usage:
    python nb2py.py NOTEBOOK.ipynb [--out OUT.py] [--strip]
"""
from __future__ import annotations
import json
import os
import re
import sys
import argparse

MAGIC = re.compile(r"^\s*[%!?]")
IMPORT = re.compile(r"^\s*(import\s|from\s+\S+\s+import\b)")


def convert(nb_path: str, strip: bool = False) -> str:
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    name = os.path.basename(nb_path)
    out: list[str] = []
    out.append(f"# {'='*70}")
    out.append(f"# Auto-converted from notebook: {name}")
    if strip:
        out.append("# EXERCISE SKELETON — code cells stripped to TODOs.")
        out.append("# Markdown narration and imports are kept as guidance. Fill in the rest.")
    else:
        out.append("# Reference solution (full code from the notebook).")
    out.append("# '# %%' markers let you run this cell-by-cell in VS Code / Jupyter.")
    out.append(f"# {'='*70}")
    out.append("")

    for i, cell in enumerate(nb.get("cells", [])):
        ct = cell.get("cell_type")
        src = cell.get("source", [])
        if isinstance(src, list):
            text = "".join(src)
        else:
            text = src

        if ct == "markdown":
            if not text.strip():
                continue
            out.append("# %% [markdown]")
            for line in text.splitlines():
                out.append("# " + line if line.strip() else "#")
            out.append("")
        elif ct == "code":
            if not text.strip():
                continue
            out.append(f"# %%")
            lines = text.split("\n")
            if strip:
                prev_todo = False
                for l in lines:
                    s = l.strip()
                    if not s:
                        continue
                    if IMPORT.match(l) or s.startswith("#"):
                        out.append(l)
                        prev_todo = False
                    else:
                        if not prev_todo:
                            out.append("# TODO: implement this cell (see solution / lecture video)")
                            prev_todo = True
                out.append("")
            else:
                for l in lines:
                    if MAGIC.match(l):
                        out.append("# " + l + "   # [notebook magic — commented out]")
                    else:
                        out.append(l)
                out.append("")

    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--out", default=None)
    ap.add_argument("--strip", action="store_true")
    args = ap.parse_args()

    code = convert(args.notebook, strip=args.strip)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(code)
        # validate solution output parses (skeletons with TODO-only cells also parse)
        import ast
        try:
            ast.parse(code)
        except SyntaxError as e:
            print(f"WARNING: {args.out} did not parse cleanly: {e}", file=sys.stderr)
    else:
        sys.stdout.write(code)


if __name__ == "__main__":
    main()
