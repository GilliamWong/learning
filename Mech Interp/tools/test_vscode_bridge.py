"""Check persistence and notebook/reader behavior without touching learner progress."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import nbformat
import pymupdf

from vscode_bridge import WORKSPACE, handle, environment_for
from workbench.state import Progress, new_notebook


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vscode-bridge-", dir=WORKSPACE / ".runtime")
        self.root = Path(self.temp.name)
        course = json.loads((WORKSPACE / "curriculum.json").read_text(encoding="utf-8"))
        course["resources"] = [{"id": "paper", "title": "Reader check", "author": "Test", "type": "Paper", "module": "01",
                               "url": "https://example.com/paper", "local": "library/check.pdf", "target": "Check pages."}]
        (self.root / "curriculum.json").write_text(json.dumps(course), encoding="utf-8")
        (self.root / "LEARNING_PLAN.md").write_text("# Plan\n\n[Module](modules/test.ipynb)\n\n<script>unsafe()</script>", encoding="utf-8")
        (self.root / "library").mkdir()
        with pymupdf.open() as document:
            for index in range(3):
                page = document.new_page()
                page.insert_text((72, 72), f"Page {index + 1}")
            document.save(self.root / "library/check.pdf")
        shutil.copytree(WORKSPACE / "templates", self.root / "templates")

    def tearDown(self):
        self.temp.cleanup()

    def test_browser_and_vscode_updates_preserve_each_other(self):
        browser = Progress(self.root)
        browser.note("session", "A note already written in JupyterLab")
        handle(self.root, "complete", {"module": "01", "task": "batch", "value": True})
        browser.complete("01", "center", True)
        handle(self.root, "note", {"key": "module-03", "value": "Trace attention dimensions"})
        result = handle(self.root, "snapshot", {})["progress"]
        self.assertEqual(set(result["completed"]["01"]), {"batch", "center"})
        self.assertEqual(result["notes"]["session"], "A note already written in JupyterLab")
        handle(self.root, "complete", {"module": "01", "task": "batch", "value": False})
        self.assertEqual(browser.read()["completed"]["01"], ["center"])
        self.assertTrue((self.root / "progress.json.bak").is_file())

    def test_concurrent_frontend_writes_are_not_lost(self):
        def write(index):
            result = subprocess.run([sys.executable, "-X", "utf8", str(WORKSPACE / "tools/vscode_bridge.py"), "note", "--root", str(self.root)],
                                    input=json.dumps({"key": f"note-{index}", "value": str(index)}), text=True, capture_output=True, check=True)
            self.assertTrue(json.loads(result.stdout)["ok"])
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(write, range(8)))
        notes = Progress(self.root).read()["notes"]
        self.assertEqual(len(notes), 8)

    def test_corrupt_progress_is_preserved(self):
        path = self.root / "progress.json"
        path.write_text("{damaged", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            handle(self.root, "note", {"key": "session", "value": "Must not overwrite"})
        self.assertEqual(path.read_text(), "{damaged")

    def test_paper_pages_share_the_reader_checkpoint(self):
        Progress(self.root).note("paper-page-paper", "2")
        result = handle(self.root, "paper", {"id": "paper", "mode": "Text"})
        self.assertEqual(result["page"], 2)
        self.assertIn("Page 2", result["text"])
        result = handle(self.root, "paper", {"id": "paper", "page": 99})
        self.assertEqual(result["page"], 3)
        self.assertGreater(len(result["image"]), 100)
        self.assertEqual(Progress(self.root).read()["notes"]["paper-page-paper"], "3")

    def test_new_notebooks_preserve_templates_and_existing_work(self):
        template = self.root / "templates/Experiment.ipynb"
        before = template.read_bytes()
        first = handle(self.root, "create", {"kind": "experiment"})["path"]
        second = handle(self.root, "create", {"kind": "experiment"})["path"]
        self.assertNotEqual(first, second)
        self.assertEqual(template.read_bytes(), before)
        self.assertEqual(nbformat.read(self.root / first, 4).metadata.kernelspec.name, "arena")
        # The original JupyterLab call retains its existing template behavior.
        third = new_notebook("Experiment.ipynb", "experiments", root=self.root)
        self.assertEqual((self.root / third).read_bytes(), before)
        self.assertEqual(len(Progress(self.root).read()["created"]), 3)

    def test_document_and_reading_inputs(self):
        document = handle(self.root, "document", {"path": "LEARNING_PLAN.md"})
        self.assertIn('href="modules/test.ipynb"', document["html"])
        self.assertNotIn("<script>", document["html"])
        with self.assertRaises(ValueError):
            handle(self.root, "document", {"path": "../outside.md"})
        with self.assertRaises(ValueError):
            handle(self.root, "complete", {"module": "01", "task": "missing", "value": True})
        with self.assertRaises(ValueError):
            handle(self.root, "bookmark", {"title": "Bad link", "url": "javascript:alert(1)", "reason": ""})
        handle(self.root, "reading", {"id": "paper", "value": "Reading"})
        self.assertEqual(Progress(self.root).read()["reading"]["paper"], "Reading")

    def test_existing_notebooks_choose_their_correct_environment(self):
        main = environment_for(WORKSPACE, "modules/01_Tensors.ipynb")
        arena = environment_for(WORKSPACE, "arena/00_Environment_Check.ipynb")
        circuit_file = next((WORKSPACE / "arena/notebooks").glob("**/1.4.2_*"))
        circuits = environment_for(WORKSPACE, circuit_file.relative_to(WORKSPACE).as_posix())
        self.assertEqual(main["path"], str(WORKSPACE / ".venv/Scripts/python.exe"))
        self.assertEqual(arena["path"], str(WORKSPACE / "arena/.venv/Scripts/python.exe"))
        self.assertEqual(circuits["path"], str(WORKSPACE / "arena/circuits/.venv/Scripts/python.exe"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
