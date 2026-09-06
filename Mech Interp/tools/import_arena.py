"""Prepare a reviewable local copy of every notebook from the pinned ARENA snapshot.

Existing learner notebooks are never overwritten. Original files remain in arena/ARENA_3.0.
"""

import ast
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from uuid import uuid4

import nbformat
from IPython.core.inputtransformer2 import TransformerManager
from patch_arena_support import apply as apply_support_repairs, reference_repairs

ROOT = Path(__file__).resolve().parent.parent
ARENA = ROOT / "arena"
SOURCE = ARENA / "ARENA_3.0"
TRANSFORM = TransformerManager()


def local_setup(chapter, section, name):
    return "\n".join([
        "from pathlib import Path", "import runpy", "import sys",
        'workspace = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "arena" / "runtime.py").is_file())',
        'local_arena = runpy.run_path(str(workspace / "arena" / "runtime.py"))',
        f'globals().update(local_arena["setup"]({chapter!r}, {section!r}, {name!r}))',
    ])


def fix_cell(text, filename, changes):
    original = text
    text = text.replace('next(p for p in Path.cwd().parents if p.name == "ARENA_3.0")', 'ARENA_SOURCE')
    text = text.replace('next(p for p in Path.cwd().parents if p.name == repo)', 'ARENA_SOURCE')
    text = text.replace('env_path = exercises_dir / ".env"', 'env_path = ARENA_ROOT / ".env"')
    text = text.replace('env_path = Path.cwd() / ".env"', 'env_path = ARENA_ROOT / ".env"')
    if text != original:
        changes.append("Use local snapshot and credential paths.")
    original = text
    # These are malformed exercise placeholders, not answers to the exercises.
    if filename.startswith("0.5_") and "for img_real, label in progress_bar:" in text:
        marker = "                # YOUR CODE HERE - fill in the training step for generator & discriminator"
        text = text.replace(marker + "\n\n", marker + '\n                raise NotImplementedError("Implement the generator and discriminator training steps.")\n\n')
    if filename.startswith("1.3.1_") and "FEW_SHOT_PROMPT" in text:
        text = re.sub(r"^    (# Get token IDs|TRUE_ID =|FALSE_ID =)", r"\1", text, flags=re.M)
    if filename.startswith("1.4.1_") and "results = t.zeros(len(CIRCUIT" in text and "    imshow(" in text:
        begin = text.index("    imshow(")
        text = text[:begin] + "\n".join(line[4:] if line.startswith("    ") else line for line in text[begin:].splitlines())
    if filename.startswith("3.4_"):
        text = re.sub(r"^(\s*(?:system_instruction|on_page_instruction|next_step_instruction)\s*=)\s*$", r"\1 None  # TODO: implement this prompt.", text, flags=re.M)
        text = text.replace('raise NotImplementedError("You need to reimplement the instruction_refresh function\n        ")',
                            'raise NotImplementedError("You need to reimplement the instruction_refresh function")')
    if filename.startswith("4.3_") and "def compute_suppressed_attention(" in text:
        marker = "    # YOUR CODE HERE - compute attention scores, apply suppression, apply attention mask, then softmax"
        text = text.replace(marker + "\n", marker + '\n    raise NotImplementedError("Implement suppressed attention.")\n')
    if filename.startswith("4.5_") and "Extract scores from logs[0]" in text:
        first = text.index("    Implement this function using the Petri Python API.")
        last = text.index("    raise NotImplementedError()", first)
        text = text[:first] + "\n".join("    # " + line.strip() for line in text[first:last].splitlines()) + "\n" + text[last:]
        for prefix in ("    Extract scores from", "    The exact structure depends", "    Hint: logs["):
            text = text.replace(prefix, "    # " + prefix.strip())
    if text != original:
        changes.append("Repaired malformed placeholder or indentation; exercise remains unanswered.")
    before_reference = text
    if filename.startswith("0.3_"):
        text = reference_repairs(text, "optimization")
    if filename.startswith("2.5_"):
        text = text.replace("(B, 1) sampled column indices (one per game).", "(B,) sampled column indices (one per game).")
        if "solutions" in filename:
            text = reference_repairs(text, "mcts")
    if filename.startswith("3.5_"):
        text = text.replace('local_exec = "python3"', 'local_exec = sys.executable')
        if "solutions" in filename:
            text = reference_repairs(text, "control")
    if text != before_reference:
        changes.append("Apply the documented reference shape or Windows sandbox compatibility repair.")
    updated = re.sub(r"num_workers\s*=\s*[1-9][0-9]*", "num_workers=0", text)
    updated = re.sub(r"num_workers:\s*int\s*=\s*[1-9][0-9]*", "num_workers: int = 0", updated)
    if updated != text:
        changes.append("Use num_workers=0 for Windows notebook data loading.")
    text = updated
    if 'assert "ARENA_3.0" in petri_path' in text:
        text = text.replace('assert "ARENA_3.0" in petri_path', 'assert Path(petri_path).resolve().is_relative_to(ARENA_SOURCE.resolve())')
        changes.append("Validate the local Petri path against the actual snapshot folder.")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-untouched", action="store_true", help="Update only files still matching their imported checksum")
    args = parser.parse_args()
    apply_support_repairs()
    old_manifest = json.loads((ARENA / "manifest.json").read_text(encoding="utf-8")) if (ARENA / "manifest.json").exists() else {"notebooks": []}
    previous = {r["local"]: r for r in old_manifest["notebooks"]}
    revision = subprocess.check_output(["git", "-c", f"safe.directory={SOURCE}", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    records = []
    tracked = subprocess.check_output(["git", "-c", f"safe.directory={SOURCE}", "-C", str(SOURCE), "ls-files", "*.ipynb"], text=True).splitlines()
    # Supporting repositories may also have notebooks; only import ARENA's tracked files.
    for path in sorted(SOURCE / name for name in tracked if name.startswith("chapter")):
        relative = path.relative_to(SOURCE)
        chapter = relative.parts[0]
        section = Path(*relative.parts[2:-1]).as_posix()
        kind = "exercise" if "_exercises.ipynb" in path.name else "solution" if "_solutions.ipynb" in path.name else "additional"
        directory = {"exercise": "notebooks", "solution": "reference", "additional": "additional"}[kind]
        target = ARENA / directory / chapter / section / path.name
        notebook = nbformat.read(path, as_version=4)
        notebook.nbformat_minor = max(notebook.nbformat_minor, 5)
        for cell in notebook.cells:
            cell.setdefault("id", uuid4().hex[:8])
            if cell.cell_type != "code":
                cell.pop("execution_count", None)
                cell.pop("outputs", None)
        changes = []
        bootstrap = local_setup(chapter, section, path.name)
        bootstrap_added = False
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            if "IN_COLAB" in cell.source and "pip install" in cell.source and ("# Get root" in cell.source or "# Get Root" in cell.source):
                cell.source = bootstrap
                cell.metadata["tags"] = ["arena-local-setup"]
                bootstrap_added = True
                changes.append("Replaced Colab/Linux installer with the local setup cell.")
            else:
                cell.source = fix_cell(cell.source, path.name, changes)
                lines = cell.source.splitlines()
                clean = ["# Dependency installation is handled by Setup ARENA.cmd." if re.match(r"\s*[!%]pip\s+install", line) else line for line in lines]
                if clean != lines:
                    changes.append("Removed in-notebook package installation to preserve the locked environment.")
                cell.source = "\n".join(clean)
            cell.outputs = []
            cell.execution_count = None
        if not bootstrap_added:
            notebook.cells.insert(0, nbformat.v4.new_code_cell(bootstrap, metadata={"tags": ["arena-local-setup"]}))
        kernel, display_name = ("arena-circuits", "ARENA (SAE circuits)") if path.name.startswith("1.4.2_") else ("arena", "ARENA (local GPU)")
        notice = (f"**Local ARENA · {chapter.replace('_', ' ')} · {kind}**\n\n"
                  f"Use the **{display_name}** kernel selected for this notebook. Run the local setup cell first. "
                  "Package installation and Linux/Colab paths have been adapted; the original is retained in `arena/ARENA_3.0`. "
                  "Exercise blanks are intentional. API-dependent sections require your own credentials and are run manually.\n\n"
                  f"[Original notebook](https://github.com/callummcdougall/ARENA_3.0/blob/{revision}/{relative.as_posix()})")
        notebook.cells.insert(0, nbformat.v4.new_markdown_cell(notice))
        notebook.metadata["kernelspec"] = {"name": kernel, "display_name": display_name, "language": "python"}
        notebook.metadata.pop("widgets", None)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                try:
                    ast.parse(TRANSFORM.transform_cell(cell.source))
                except (SyntaxError, ValueError) as error:
                    raise RuntimeError(f"Unrepaired syntax: {relative}, cell {index}: {error}") from error
        nbformat.validate(notebook)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing = nbformat.read(target, as_version=4)
            expected_content = [(c.cell_type, c.source) for c in notebook.cells]
            existing_content = [(c.cell_type, c.source) for c in existing.cells]
            if expected_content != existing_content:
                prior = previous.get(target.relative_to(ROOT).as_posix(), {})
                unedited = hashlib.sha256(target.read_bytes()).hexdigest() == prior.get("imported_sha256")
                if not (args.update_untouched and unedited):
                    raise FileExistsError(f"Refusing to overwrite changed learner work: {target}")
                nbformat.write(notebook, target)
        else:
            nbformat.write(notebook, target)
        records.append({"source": relative.as_posix(), "local": target.relative_to(ROOT).as_posix(),
                        "chapter": chapter, "kind": kind, "title": path.stem.replace("_", " "),
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "imported_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "adaptations": sorted(set(changes)), "syntax": "passed",
                        "execution_report": "arena/validation/setup-results.json"})
    manifest = {"upstream": "https://github.com/callummcdougall/ARENA_3.0", "revision": revision,
                "imported_at": datetime.now(timezone.utc).isoformat(), "paid_api_calls": "manual", "notebooks": records}
    (ARENA / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Prepared {len(records)} notebooks: " + ", ".join(f"{sum(r['kind']==kind for r in records)} {kind}" for kind in ["exercise", "solution", "additional"]))


if __name__ == "__main__":
    main()
