# VS Code validation — 6 September 2026

[Using the VS Code workbench](VSCODE.md) · [Learning plan](LEARNING_PLAN.md)

## Checked in the real VS Code installation

- Installed the local `local-learning.mech-interp-workbench` extension through VS Code's CLI.
- You opened the course folder and confirmed that the native Learning Dashboard appeared.
- You followed the study plan/catalog link to the environment-check notebook and confirmed the GPU PASS
  message, Plotly heatmap, and CircuitsVis attention diagram inside VS Code.
- The extension's output log confirmed that it opened the environment check with **ARENA (local GPU)**.
  Jupyter's log confirmed that the intended `arena/.venv/Scripts/python.exe` kernel started successfully.
- The verified installation uses VS Code 1.136.1, Jupyter extension 2025.9.1, and Python extension 2026.4.0.

Computer control was stopped when you chose to open the folder yourself. The confirmations above came
from you and the corresponding local application logs. Later checks used temporary test state and did
not operate your VS Code window.

## Automated checks

Seven Python checks passed for the shared bridge:

- JupyterLab-style and VS Code-style updates preserve each other's notes and checkmarks.
- Eight simultaneous writes retain every note.
- Corrupt progress is left intact and reported instead of overwritten.
- PDF readers load and update the same saved page key as the existing reader.
- New notebook copies preserve templates and existing work; new native experiments select the ARENA kernel.
- Reading/task input validation and workspace-local document links behave correctly.
- Existing notebooks map to the three intended Python environments.

The real frontend script was exercised in a test DOM against the Python bridge with a temporary workspace.
Checks cover navigation, plan links, checklist persistence, saving notes before changing tabs, ARENA filters,
paper links, new experiments, retrying a failed save without losing text, advancing Continue after completing
a module, and the paper reader's pagination/text/image modes.

Five cells also executed in a fresh Jupyter kernel: the shared notebook home, a module panel, the local PDF
reader, and checks for both VS Code and JupyterLab link formats. This checks the Python widget/reader code;
it is separate from visual frontend confirmation.

The browser launcher passed a regression check under the normal Windows user permissions required to read
Jupyter's protected cookie file. Its standard CPython launcher repair remains in the shared setup script.

## Scope

No learner exercise implementations were filled or replaced, and no paid APIs or additional training runs
were executed for this port. The existing [ARENA execution boundaries](arena/VALIDATION.md) remain in force.
The standalone paper-reader UI and all individual research websites were not visually inspected in VS Code;
their local rendering/navigation behavior is covered by component tests and the editor's browser API.

The final small updates to Continue, save recovery, and portable notebook links are included in the installed
extension. Reload an already open learning window once to load that final version.

## Kernel startup follow-up

ARENA 1.1 later stalled during the VS Code startup handshake with **ipykernel 7.3.0**. The Python process
was alive and responded to a read-only kernel-info request; VS Code then completed the pending connection.
This matches a [reported ipykernel 7 hang with Jupyter extension 2025.9.1](https://github.com/microsoft/vscode-jupyter/issues/17410).

All three learning environments now explicitly pin **ipykernel 6.31.0**. The SAE-circuits environment already
used that version. Three fresh ARENA kernels connected in **0.80, 0.73, and 0.73 seconds**. Both ARENA 1.1
setup/import prefixes and the five shared-panel/reader test cells passed with the pin. Dependency checks
passed for the workbench's 113 packages, ARENA's 257 packages, and the circuit environment's 149 packages.

After restarting ARENA 1.1, you confirmed that setup completes. The native Jupyter log recorded the restart
request at 12:53:11.435 and completion at 12:53:12.013. The diagnostic did not execute exercise answers or
change your notebook work. The startup timing records are in `.runtime/kernel-start-check/ipykernel-6-results.json`.

## Reproduce the automated checks

From this workspace root:

```powershell
.venv\Scripts\python.exe -X utf8 tools\test_vscode_bridge.py
node tools\test_vscode_frontend.cjs
.venv\Scripts\python.exe -X utf8 tools\test_vscode_notebooks.py
```

The frontend test's only extra dependency is test-only:

```powershell
npm install --prefix .runtime/vscode-unit --no-save --ignore-scripts --cache .cache/npm linkedom
```

Fresh-kernel output is stored under `.runtime/vscode-validation`. The storage and frontend tests create and
remove their own temporary workspaces there; they do not write to your learning progress or answers.
