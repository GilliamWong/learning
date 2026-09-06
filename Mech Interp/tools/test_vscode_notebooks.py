"""Execute shared notebook panels/readers in a fresh kernel using VS Code link mode."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

from launch import configure_environment

ROOT = Path(__file__).resolve().parent.parent
configure_environment()
output = ROOT / ".runtime/vscode-validation"
output.mkdir(parents=True, exist_ok=True)
setup = f'''import os, sys
from pathlib import Path
os.environ["MECH_INTERP_FRONTEND"] = "vscode"
sys.path.insert(0, {str(ROOT)!r})
from IPython.display import display
from workbench.ui import home, lesson_panel, file_link, web_link
from workbench.reader import paper_reader
assert "vscode://local-learning.mech-interp-workbench/open?" in file_link("LEARNING_PLAN.md", "Plan")
'''
notebook = nbformat.v4.new_notebook(cells=[
    nbformat.v4.new_code_cell(setup),
    nbformat.v4.new_code_cell("display(home())"),
    nbformat.v4.new_code_cell("display(lesson_panel('03'))"),
    nbformat.v4.new_code_cell("display(paper_reader('induction'))"),
    nbformat.v4.new_code_cell('''os.environ.pop("MECH_INTERP_FRONTEND")
assert "markdownviewer:open" in file_link("LEARNING_PLAN.md", "Plan")
assert "docmanager:open" in file_link("modules/03_Transformer_Internals.ipynb", "Module")
print("Shared panels, paper reader, and both link modes passed.")''')
])
NotebookClient(notebook, kernel_name="python3", timeout=120, resources={"metadata": {"path": str(ROOT)}}).execute()
nbformat.write(notebook, output / "shared-panels.ipynb")
record = {"status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(),
          "coverage": "Five fresh-kernel cells: shared home, module panel, local PDF reader, VS Code URI links and JupyterLab links. Frontend visual rendering is a separate check."}
(output / "shared-panels.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
