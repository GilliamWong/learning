# Maintaining this workspace

The learning content is the product. Keep the supporting code small.

- Edit prose and exercises directly in the notebooks. No generator is required for later edits.
- `curriculum.json` is the ordered module and resource index. Each task has a stable ID; changing an ID resets its checkbox identity.
- Add a reading to the `resources` list with its title, author, URL, module, assigned sections (`target`), and reading depth.
- A resource may include `local` for a project-relative PDF and `web` for an alternative web edition.
- Add a module by creating a notebook and adding its file, title, and tasks to `curriculum.json`. Call `lesson_panel("ID")` from its setup cell.
- Personal bookmarks can be added from **Your work** without editing JSON.
- `workbench/ui.py` draws the notebook widgets. `state.py` saves progress atomically and locks across kernels.
- `workbench/checks.py` checks the original opening exercises. Checks never mark tasks complete automatically.
- `workbench/reader.py` renders local paper pages with PyMuPDF. It avoids dependence on the browser's native PDF plug-in.
- Internal links use Jupyter's documented `docmanager:open` command through its rich-output renderer.
- `tools/launch.py` runs one authenticated Jupyter server on the loopback interface. It reuses a running instance and stops only this project's server through Jupyter's API.

## Environment

Dependencies are in `pyproject.toml`; exact resolved versions are in `uv.lock`.
`uv sync --frozen --cache-dir .cache/uv` recreates the tested environment.
The opening environment uses CPU PyTorch. Add transformer-specific dependencies when authoring the next full local module.
The official ARENA notebooks have their own setup and can be opened using their Colab exercise links in the meantime.
Course notebooks reviewed during setup were signed using Jupyter's standard notebook trust mechanism.
The launcher does not automatically trust newly imported notebooks.

## Progress and personal work

`progress.json` contains task completion, reading status, notes, bookmarks, and created notebook paths.
Its previous successful version is copied to `progress.json.bak`. Invalid progress is never silently replaced.
Notes typed in widgets save automatically; notebook code and Markdown answers use Jupyter's normal save mechanism.
Runtime data, environment files, and personal progress are excluded from Git.

## Source policy

Keep original sources identifiable. The first two local exercise sets and connecting explanations were written for this learner;
they are not claimed to be official ARENA exercises. ARENA resources open at the source. Public paper PDFs are stored for local study,
with download URLs and hashes in `library/provenance.json`. Web research articles remain linked to the publisher.
Do not silently substitute an educational approximation for a paper's original experimental setup.

## Validation

The initial delivery checks notebook structure, executes the unfilled lesson paths without errors,
exercises the contracts using completed temporary candidates, checks that incorrect candidates fail,
and verifies persistent state and the actual Jupyter UI. See `VALIDATION.md` for the recorded result.
