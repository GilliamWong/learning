"""Small JSON bridge for the VS Code view; progress still uses workbench.state."""

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))
from workbench.state import Progress, new_notebook

READING_STATES = ["Not started", "Reading", "Assigned sections reviewed", "Revisit"]


def local_path(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Choose a workspace file.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("That file is outside this learning workspace.")
    return path


def course_at(root):
    return json.loads((root / "curriculum.json").read_text(encoding="utf-8"))


def markdown_document(root, relative):
    import mistune
    path = local_path(root, relative)
    if path.suffix.lower() != ".md":
        raise ValueError("Choose a Markdown reading guide.")
    markdown = mistune.create_markdown(escape=True, plugins=["table", "strikethrough"])
    return {"path": path.relative_to(root).as_posix(), "title": path.stem,
            "html": markdown(path.read_text(encoding="utf-8"))}


def environment_for(root, relative):
    path = local_path(root, relative)
    if path.suffix != ".ipynb":
        raise ValueError("Choose a notebook.")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    kernel = notebook.get("metadata", {}).get("kernelspec", {}).get("name", "python3")
    if kernel == "arena-circuits" or path.name.startswith("1.4.2_"):
        directory, label = "arena/circuits", "ARENA (SAE circuits)"
    elif kernel == "arena" or path.is_relative_to(root / "arena"):
        directory, label = "arena", "ARENA (local GPU)"
    else:
        directory, label = ".", "Workbench (Python)"
    interpreter = root / directory / ".venv/Scripts/python.exe"
    if not interpreter.is_file():
        raise FileNotFoundError(f"Missing {label} environment. Run the appropriate learning setup launcher.")
    return {"path": str(interpreter), "id": str(interpreter), "label": label}


def snapshot(root):
    manifest_path = root / "arena/manifest.json"
    arena = json.loads(manifest_path.read_text(encoding="utf-8"))["notebooks"] if manifest_path.exists() else []
    return {"course": course_at(root), "progress": Progress(root).read(), "arena": arena,
            "plan": markdown_document(root, "LEARNING_PLAN.md"), "readingStates": READING_STATES}


def paper_page(root, payload):
    import pymupdf
    resources = {r["id"]: r for r in course_at(root)["resources"]}
    resource = resources[payload["id"]]
    path = local_path(root, resource["local"])
    if path.suffix.lower() != ".pdf":
        raise ValueError("This resource is not a local PDF.")
    store = Progress(root)
    key = f"paper-page-{resource['id']}"
    with pymupdf.open(path) as document:
        count = document.page_count
        saved = store.read().get("notes", {}).get(key, "1")
        try:
            requested = int(payload.get("page", saved))
        except (ValueError, TypeError):
            requested = 1
        page = max(1, min(count, requested))
        result = {"resource": resource, "page": page, "count": count,
                  "text": document[page - 1].get_text()}
        if payload.get("mode", "Page") == "Page":
            pixmap = document[page - 1].get_pixmap(dpi=135, alpha=False)
            result["image"] = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
    if "page" in payload:
        store.note(key, str(page))
    return result


def handle(root, action, payload):
    root = Path(root).resolve()
    store = Progress(root)
    if action == "snapshot":
        return snapshot(root)
    if action == "document":
        return markdown_document(root, payload["path"])
    if action == "environment":
        return environment_for(root, payload["path"])
    if action == "paper":
        return paper_page(root, payload)
    if action == "complete":
        module = next(m for m in course_at(root)["modules"] if m["id"] == payload["module"])
        if payload["task"] not in {t["id"] for t in module["tasks"]} or type(payload["value"]) is not bool:
            raise ValueError("Unknown task or completion value.")
        store.complete(payload["module"], payload["task"], payload["value"])
    elif action == "reading":
        if payload["id"] not in {r["id"] for r in course_at(root)["resources"]} or payload["value"] not in READING_STATES:
            raise ValueError("Unknown resource or reading status.")
        store.reading(payload["id"], payload["value"])
    elif action == "note":
        if not isinstance(payload["key"], str) or not isinstance(payload["value"], str):
            raise ValueError("Notes must be text.")
        store.note(payload["key"], payload["value"])
    elif action == "bookmark":
        parsed = urlparse(payload["url"])
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or not payload["title"].strip():
            raise ValueError("Add a title and a complete http or https address.")
        record = {key: payload[key].strip() for key in ["title", "url", "reason"]}
        store.update(lambda data: data.setdefault("bookmarks", []).append(record))
    elif action == "create":
        choices = {"experiment": ("Experiment.ipynb", "experiments"), "paper": ("Paper_Notes.ipynb", "notes")}
        template, folder = choices[payload["kind"]]
        relative = new_notebook(template, folder, root=root)
        if payload["kind"] == "experiment":
            # New experiment copies start with the existing GPU environment; the template stays unchanged.
            path = root / relative
            notebook = json.loads(path.read_text(encoding="utf-8"))
            notebook.setdefault("metadata", {})["kernelspec"] = {
                "name": "arena", "display_name": "ARENA (local GPU)", "language": "python"}
            path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"path": relative, "progress": store.read()}
    else:
        raise ValueError(f"Unknown action: {action}")
    return {"progress": store.read(), "savedAt": datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action")
    parser.add_argument("--root", type=Path, default=WORKSPACE)
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        print(json.dumps({"ok": True, "result": handle(args.root, args.action, payload)}, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
