"""File-backed progress. Read-modify-write under one lock across notebook kernels."""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

ROOT = Path(__file__).resolve().parent.parent


def curriculum():
    return json.loads((ROOT / "curriculum.json").read_text(encoding="utf-8"))


class Progress:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.path = self.root / "progress.json"
        self.lock = FileLock(str(self.root / ".runtime" / "progress.lock"), timeout=5)
        (self.root / ".runtime").mkdir(parents=True, exist_ok=True)

    def _read(self):
        if not self.path.exists():
            return {"version": 1, "completed": {}, "notes": {}, "reading": {}, "bookmarks": [], "created": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("version") != 1:
                raise ValueError("Unsupported progress version")
            return data
        except (ValueError, TypeError, AttributeError) as error:
            raise RuntimeError("Could not read progress.json. It has been left untouched. A previous copy may be in progress.json.bak.") from error

    def read(self):
        with self.lock:
            return self._read()

    def update(self, edit):
        with self.lock:
            data = self._read()
            edit(data)
            data["updated"] = datetime.now().isoformat(timespec="seconds")
            tmp = self.root / ".runtime" / f"progress-{uuid4().hex}.tmp"
            try:
                tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                if self.path.exists():
                    shutil.copy2(self.path, self.root / "progress.json.bak")
                os.replace(tmp, self.path)
            finally:
                tmp.unlink(missing_ok=True)

    def complete(self, module, task, value):
        def edit(data):
            done = set(data.setdefault("completed", {}).get(module, []))
            done.add(task) if value else done.discard(task)
            data["completed"][module] = sorted(done)
        self.update(edit)

    def note(self, key, value):
        self.update(lambda data: data.setdefault("notes", {}).__setitem__(key, value))

    def reading(self, key, value):
        self.update(lambda data: data.setdefault("reading", {}).__setitem__(key, value))


def new_notebook(template, folder):
    """Copy a blank template without modifying it or overwriting existing work."""
    if (template, folder) not in {("Experiment.ipynb", "experiments"), ("Paper_Notes.ipynb", "notes")}:
        raise ValueError("Unknown template")
    target = ROOT / folder
    target.mkdir(exist_ok=True)
    filename = f"{datetime.now():%Y-%m-%d_%H%M%S}_{uuid4().hex[:4]}_{template}"
    output = target / filename
    shutil.copyfile(ROOT / "templates" / template, output)
    relative = output.relative_to(ROOT).as_posix()
    Progress().update(lambda data: data.setdefault("created", []).append(relative))
    return relative
