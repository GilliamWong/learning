# Maintaining this workspace

The learning content is the product. Keep the supporting code small.

- Edit prose and exercises directly in the notebooks. No generator is required for later edits.
- `curriculum.json` is the ordered module and resource index. Each task has a stable ID; changing an ID resets its checkbox identity.
- Add a reading to the `resources` list with its title, author, URL, module, assigned sections (`target`), and reading depth.
- A resource may include `local` for a project-relative PDF, `reader` for its notebook reader, `notebook` for local exercises, and `web` for an alternative web edition.
- Add a module by creating a notebook and adding its file, title, and tasks to `curriculum.json`. Call `lesson_panel("ID")` from its setup cell.
- Personal bookmarks can be added from **Your work** without editing JSON.
- `workbench/ui.py` draws the notebook widgets. `state.py` saves progress atomically and locks across kernels.
- `workbench/checks.py` checks the original opening exercises. Checks never mark tasks complete automatically.
- `workbench/reader.py` renders local paper pages with PyMuPDF. It avoids dependence on the browser's native PDF plug-in.
- Internal links use Jupyter's `docmanager:open` or `markdownviewer:open` commands through its rich-output renderer.
- The private Jupyter settings default Markdown files to preview; right-click a file and choose an editor to edit it. Existing user preferences are preserved.
- `tools/launch.py` runs one authenticated Jupyter server on the loopback interface. It reuses a running instance and stops only this project's server through Jupyter's API.

## Environment

Dependencies are in `pyproject.toml`; exact resolved versions are in `uv.lock`.
`uv sync --frozen --cache-dir .cache/uv` recreates the tested environment.
The opening environment uses CPU PyTorch. ARENA has a separate pinned GPU environment in
`arena/pyproject.toml` and `arena/uv.lock`. **Setup ARENA.cmd** installs it and registers a kernel
only in this workbench's Jupyter data folder. `arena/circuits/` holds the isolated versions needed by
notebook 1.4.2 and its automatically selected **ARENA (SAE circuits)** kernel. Both setup scripts use standard CPython venv launchers
to avoid the observed Windows Application Control rejection of the small uv launcher.
Course notebooks reviewed during setup were signed using Jupyter's standard notebook trust mechanism.
The launcher does not automatically trust newly imported notebooks.

## Progress and personal work

`progress.json` contains task completion, reading status, notes, bookmarks, and created notebook paths.
Its previous successful version is copied to `progress.json.bak`. Invalid progress is never silently replaced.
Notes typed in widgets save automatically; notebook code and Markdown answers use Jupyter's normal save mechanism.
Runtime data, environment files, and personal progress are excluded from Git.

## Source policy

Keep original sources identifiable. The first two local exercise sets and connecting explanations were written for this learner;
they are not claimed to be official ARENA exercises. ARENA notebooks are imported from the pinned official repository,
with source links and per-file hashes in `arena/manifest.json`. Public paper PDFs are stored for local study,
with download URLs and hashes in `library/provenance.json`. Web research articles remain linked to the publisher.
Do not silently substitute an educational approximation for a paper's original experimental setup.

## Validation

The initial delivery checks notebook structure, executes the unfilled lesson paths without errors,
exercises the contracts using completed temporary candidates, checks that incorrect candidates fail,
and verifies persistent state and the actual Jupyter UI. See `VALIDATION.md` for the recorded result.

## ARENA maintenance

The local course snapshot is `arena/ARENA_3.0`; learner copies are under `arena/notebooks` and
references under `arena/reference`. The source checkout and caches are ignored by the parent Git
repository. `tools/import_arena.py` prepares the copies and records adaptations. It refuses to replace
changed learner work. `--update-untouched` permits updates only while the previous import hash still matches.

`tools/patch_arena_support.py` contains the small documented upstream support-code repairs.
`tools/fetch_arena_dependencies.py` fetches the supporting public repositories recorded in
`arena/supporting-repositories.json`; it restores missing checkouts at their saved revisions and preserves
existing checkouts. Petri is pinned to v2.0.0 to keep the API expected by the notebook. Large Git LFS artifacts remain deferred.
`tools/build_arena_catalog.py` rebuilds the catalog from the manifest and validation reports.

`tools/validate_arena.py` checks every setup/import prefix in a fresh kernel.
`tools/run_arena_reference_notebook.py` runs a selected full reference notebook in order and stores its
outputs separately. `tools/check_arena_reference.py` checks available numerical helpers; it cannot
replace sequential execution where the notebook builds required state in earlier cells.
The recorded coverage and manual/deferred work are in [arena/VALIDATION.md](arena/VALIDATION.md).
These checks do not fill exercise answers or make paid API calls.

## VS Code companion

`vscode-extension/` contains the native dashboard, its stylesheet, and a small VS Code integration module.
There is no frontend build framework or production npm dependency. The bridge in `tools/vscode_bridge.py`
uses `workbench.state.Progress` for the same locked/atomic progress writes and PyMuPDF for local paper pages.
The study plan is rendered from `LEARNING_PLAN.md`; the curriculum and ARENA manifest remain the indexes.

`tools/package_vscode_extension.py` creates the local VSIX with Python's standard library. The VS Code launcher
installs it only when the source hash changes or it is missing. It does not publish anything to a marketplace.
After a source update, rerun **Start Learning in VS Code.cmd** and reload an already open learning window.

Notebook opening uses the Jupyter extension's exported `openNotebook` API to select an existing interpreter;
the fallback reports the exact interpreter for the normal kernel picker. Browser sources use VS Code's
integrated-browser command. These two integrations are isolated in `vscode-extension/extension.js`.
`.vscode/learning.env` identifies the native notebook frontend. Browser startup clears that marker so
JupyterLab keeps its own document links. Notebook helper colors support both editors' theme variables.

`tools/prepare_workbench.ps1` holds the shared environment preparation; the two launchers then open their
respective interfaces. The existing Windows Application Control launcher repair is preserved.

Run `tools/test_vscode_bridge.py` with the workbench Python for the storage/reader tests, and
`tools/test_vscode_notebooks.py` for fresh-kernel execution of shared panels. Frontend interaction tests use
`node tools/test_vscode_frontend.cjs`; their test-only DOM dependency can be installed with
`npm install --prefix .runtime/vscode-unit --no-save --ignore-scripts --cache .cache/npm linkedom`.
All mutable test state lives in temporary folders under `.runtime`.
