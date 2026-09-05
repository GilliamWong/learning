# Mechanistic Interpretability Workbench

A local course for moving from ML concepts to independent experiments.

## Start

Double-click **Start Learning.cmd** in this folder. JupyterLab opens the home notebook.
Click **Open / refresh workbench** to activate the home page. Then choose **Continue learning**.
The menu command **Run > Run All Cells** is an alternative.
In a lesson, use **Shift+Enter** to run a cell and move to the next one.

The first launch installs the locked Python environment. Later launches reuse it.
The first two modules use the CPU and need no accounts, paid services, model downloads, or GPU.

Your code lives in `modules/`. Progress and notes are saved automatically in `progress.json`.
Use **File > Save Notebook** (or Ctrl+S) to save code and written notebook answers.
Checkmarks record your own assessment; passing a code check does not automatically mark a lesson complete.

## What is here

- `00_Start_Here.ipynb`: home, roadmap, reading library, bookmarks, and session notes.
- `modules/01_Tensors.ipynb`: original guided exercises with checks and optional hints.
- `modules/02_Train_a_Small_Model.ipynb`: a small classification experiment to implement and investigate.
- `modules/03_Transformer_Internals.ipynb`: a reading and implementation guide into ARENA 1.1.
- `modules/04_Induction_Heads.ipynb`: a paper-reading and experiment plan into ARENA 1.2.
- `modules/05_Causal_Circuits.ipynb`: a guide to IOI and a first replication.
- `modules/06_Independent_Experiments.ipynb`: a scaffold for choosing and documenting your own question.
- `library/`: source index, local papers where available, and reading notes.
- `library/readers/`: notebook paper readers with page navigation and selectable text; each remembers your last page.
- `templates/`: reusable paper and experiment notebooks.

Modules 1–2 are complete local exercise sets. Modules 3–6 are guided next stages that link to the
official exercises and identify reading targets; their transformer environments and full
replications are not provisioned yet. The home page labels this distinction.

## Keep it simple

This is JupyterLab plus ordinary notebooks, a small Python helper, and a JSON curriculum.
There is no custom web server, database, build pipeline, hosted service, or background scheduler.
Jupyter itself runs a local server while you work. Use **Stop Learning.cmd** to stop it.

If an activation button is clicked while a new kernel is still connecting, wait for **Idle** in the bottom
status bar and click again. You only need to activate a home or reader page once per fresh kernel.

See `MAINTAINING.md` for the file map and how to add a module or resource.
