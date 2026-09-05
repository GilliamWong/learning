# Initial validation — 2026-09-04

## Execution

- Validated and executed the home notebook, six module notebooks, and two blank templates in fresh Python kernels.
- Unfilled exercise functions remain unfilled. Running the supplied notebook paths reports unfinished work without crashing.
- Executed temporary completed candidates for all seven exercise contracts; all passed.
- Confirmed that each contract rejects an incorrect candidate.
- Ran the Module 2 harness with a temporary reference classifier and training implementation. Its fixed baseline reached 1.0 validation accuracy on the 200 generated validation examples. This validates the teaching harness; it is not a learner result or research finding.
- Validated and executed all three paper-reader notebooks. The source PDFs have 61, 25, and 62 pages respectively; the first and last pages rendered successfully.
- Temporary completed solution notebooks were never written into the course.

## Persistence and interface

- Checked read/modify/write updates through two independent progress readers, unchecking tasks, previous-version backup, and preservation of an invalid progress file. FileLock serializes writes across kernels.
- Verified the real JupyterLab UI: home activation, native document navigation, progress checkboxes updating the next action, and a note surviving a fresh server/kernel session.
- Verified creation of a fresh experiment from its template without modifying the template.
- The embedded browser's native PDF pane was blank. A small notebook reader now renders papers as pages and offers selectable text; both modes and page navigation were verified in the UI.
- Verified stop and restart of this project's server. The launcher uses a loopback address and an authentication token.

## Scope

Modules 1–2 are complete local exercise sets. Modules 3–6 are reading and implementation guides into the original resources,
plus research scaffolds. Their full transformer environments, model downloads, and paper replications have not been executed here.
Web articles open at the publisher; three selected papers are available locally through the reader.
