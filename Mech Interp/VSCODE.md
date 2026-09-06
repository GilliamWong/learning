# Learning in VS Code

Open this **Mech Interp** folder in VS Code. The **Learning Dashboard** opens automatically. You can also
click **Learning** in the status bar, use the Learning sidebar, or run **Learning: Open Dashboard** from
the Command Palette (**Ctrl+Shift+P**).

**Start Learning in VS Code.cmd** installs or updates the local companion extension and opens the folder.
You can keep opening the folder yourself once it is installed.

## Your normal workflow

1. Open **Continue** or **Study plan**. The plan contains the exercise, reading, paper, and experiment order.
2. Open an exercise from the dashboard. It opens as a native VS Code notebook with its existing Python
   environment selected. Run the local setup cell, then work through the exercises with **Shift+Enter**.
3. Open readings alongside it. Local PDFs have a page/text reader; web articles open in VS Code's integrated browser.
4. Use **Course checklists** to record milestones and your next step. Checkmarks and dashboard notes save automatically.
5. Use **Your work** to create an experiment or paper-note notebook. Save notebook code and written answers with **Ctrl+S**.

Given your completed warm-up code, **Continue** starts with Module 3 / ARENA 1.1. It advances through Modules
3–6 as their checklists are completed. The study plan explains where to insert the first small experiment,
Neel's guidance, and the optional representations branch. The earlier checklists remain available for the
warm-up reflections.

## Everything stays in this workspace

VS Code and JupyterLab share the same notebooks, `curriculum.json`, `LEARNING_PLAN.md`, and `progress.json`.
There is no separate copy of your answers. Dashboard notes and checkmarks use the same locked, atomic writes
as the browser workbench. Local paper readers remember the same page in both interfaces.

Refresh the dashboard or reopen it from the Learning sidebar to see changes made in another interface.
Prefer editing a particular notebook in one interface at a time, so two unsaved editor buffers do not compete.

## Notebooks and environments

The dashboard chooses the environment when it opens a notebook. If you open a file directly through the
Explorer and VS Code asks for a kernel, use **Select Kernel → Python Environments**:

| Notebook | Interpreter inside this workspace |
|---|---|
| Original modules and notebook paper readers | `.venv/Scripts/python.exe` |
| Most ARENA notebooks and new experiments | `arena/.venv/Scripts/python.exe` |
| ARENA 1.4.2 SAE circuits | `arena/circuits/.venv/Scripts/python.exe` |

Use **Check notebook setup** on the dashboard, or [the environment-check notebook](arena/00_Environment_Check.ipynb),
to check the GPU and interactive charts. It makes no paid API calls.

The native Learning Dashboard replaces the JupyterLab-only activation button in `00_Start_Here.ipynb`.
The shared module panels and notebook paper readers also support VS Code links and colors.

## If you are updating an already open window

After installing an updated companion extension, save open notebooks, then run **Developer: Reload Window**
once from the Command Palette to load the new version. VS Code's normal extension update notification may
offer the same reload action.

If automatic kernel selection is unavailable in a future Jupyter extension version, the notebook still opens
and the workbench shows the exact interpreter to select. Use **Setup ARENA.cmd** if an ARENA environment
needs restoring.

## What was checked

The native dashboard loaded on this computer. You confirmed that the environment-check notebook produced
the GPU PASS message, Plotly heatmap, and CircuitsVis attention diagram in VS Code. The Jupyter log also
confirmed the intended ARENA interpreter started successfully.

Additional checks cover shared progress writes, concurrent updates, save-error recovery, new notebook copies,
ARENA filters, local links, PDF page checkpoints, and execution of the shared notebook panels/readers.
See [VS Code validation](VSCODE_VALIDATION.md) for the precise coverage.

The [ARENA validation limits](arena/VALIDATION.md) still apply: paid APIs remain manual, and some later
large-model, long-training, or Linux/multi-GPU work needs additional resources.
