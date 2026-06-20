# LLMs From Scratch — Coding Practice

A hands-on workspace for re-implementing the core of modern deep learning and
LLMs, following two of the best "from scratch" courses. For every file you fill
in you get **two difficulty tiers** plus **the full reference solution**.

Each fillable file starts with the same short comment describing *what the file
should do, its inputs/outputs, and its behavior* — and nothing more. From there:

- **`bare/`** — just that comment and the necessary imports. You write
  everything else yourself, **including the class/function signatures**. Hardest.
- **`with-headers/`** — the same comment, the imports, and the method/function
  headers with `raise NotImplementedError` bodies — a first pass so you don't get
  stuck. This tier is runnable: tests collect immediately and go green as you
  implement.
- **`*-solutions/`** — the authors' real, working code, to check your answers.

Suggested flow: try a file in **`bare/`**; if you get stuck on structure, peek at
the same file in **`with-headers/`**; verify against **`*-solutions/`**.

---

## Resources

Just the two authors for now.

**Andrej Karpathy — _Neural Networks: Zero to Hero_**
- Course + lecture videos (playlist): https://karpathy.ai/zero-to-hero.html
- `nn-zero-to-hero` (lecture notebooks): https://github.com/karpathy/nn-zero-to-hero
- `micrograd` (scalar autograd engine): https://github.com/karpathy/micrograd
- `makemore` (character-level language models): https://github.com/karpathy/makemore
- `minGPT` (minimal GPT): https://github.com/karpathy/minGPT

**Sebastian Raschka — _Build a Large Language Model (From Scratch)_**
- Book: https://www.manning.com/books/build-a-large-language-model-from-scratch
- Code repo: https://github.com/rasbt/LLMs-from-scratch

---

## Folder layout

```
exercises/
├── README.md                ← you are here
├── requirements.txt
├── .venv/                   ← ready-to-use Python 3.12 env (torch, etc.)
│
├── karpathy/
│   ├── bare/                ← comment + imports only (write everything)
│   │   ├── micrograd/  makemore/  minGPT/  nn-zero-to-hero/
│   └── with-headers/        ← + method headers with NotImplementedError (runnable)
│       ├── micrograd/  makemore/  minGPT/  nn-zero-to-hero/
├── karpathy-solutions/      ← full code + every notebook converted to .py
│
├── raschka/
│   ├── bare/                ← ch02–07 = comment + imports only
│   │   └── pkg/  conftest.py  pyproject.toml  data/
│   └── with-headers/        ← ch02–07 = method headers (runnable)
│       └── pkg/  conftest.py  pyproject.toml  data/
├── raschka-solutions/       ← the complete LLMs-from-scratch repo
│
└── _tools/                  ← the generators (see "How this was built")
```

Both tiers are complete, self-contained copies — support files (tests, datasets,
trainers, tokenizers, reference model variants) are identical in each, so the
**`with-headers/`** tree runs as-is. Only the fillable files differ between tiers.

---

## Setup

A virtual environment is **already created** at `.venv/` (Python 3.12, PyTorch
installed). Activate it:

```bash
cd "exercises"
source .venv/bin/activate
```

To recreate it elsewhere:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Why 3.12? Raschka's package pins `requires-python >=3.10,<3.13`, and PyTorch
> has no 3.14 wheels yet. 3.11 or 3.12 are safe.

---

## How to work

Find your next task:
```bash
grep -rln "TODO: implement" karpathy/with-headers raschka/with-headers   # files with stubs
```
Open the matching file in `bare/` (or `with-headers/` if you'd like the
signatures), implement it, and run that component's tests. The commands below use
`with-headers/` because it's runnable immediately — once you've written the
required names in `bare/`, the same commands work there too.

> Note on `bare/`: because the class/function names don't exist until you write
> them, the tests can't even import until you do. That's the point of the tier —
> the `with-headers/` tree is where tests run from the start.

---

## Track 1 — Karpathy

### `micrograd` — a scalar-valued autograd engine
Implement `micrograd/micrograd/engine.py` (the `Value` class + `.backward()`,
import path `micrograd.engine`) and `micrograd/micrograd/nn.py` (`Neuron`,
`Layer`, `MLP`). Test against PyTorch:
```bash
cd karpathy/with-headers/micrograd && PYTHONPATH=. pytest test/test_engine.py -q
```
Then run `python demo.py` (trains a 2-layer MLP on the moons dataset).

### `makemore` — character-level language models
Implement the models in `makemore.py` — `Bigram`, then `MLP`, `RNN`/`GRU`, and
the `Transformer` (plus `CharDataset` and `generate`). The data pipeline,
training loop, and CLI are provided in `with-headers/`:
```bash
cd karpathy/with-headers/makemore
python makemore.py --type bigram --max-steps 2000 --device cpu   # then mlp, rnn, transformer
```

### `minGPT` — a minimal GPT
Implement `mingpt/model.py` (`CausalSelfAttention`, `Block`, `GPT`). The trainer,
tokenizer, config, and example projects are provided. Validate with
`python demo.py` (tiny GPT on a sorting task). `tests/test_huggingface_import.py`
additionally checks against HuggingFace GPT-2 (`pip install transformers`).

### `nn-zero-to-hero` — the lectures, as worksheets
The lecture notebooks are converted to `.py` (`# %%` cell markers). In
`with-headers/` the markdown narration, imports and Karpathy's comments are kept
with the code stripped to `# TODO`s; in `bare/` you get just the lecture's goal
and imports. Full converted lectures live in `karpathy-solutions/nn-zero-to-hero/`.

**Suggested order:** micrograd → makemore (bigram → MLP → RNN → transformer) →
the lectures alongside the videos → minGPT.

---

## Track 2 — Raschka

The exercise surface is the installable `llms_from_scratch` package. The core
teaching classes are the fillable files; reference/advanced variants (`*Fast`
models, KV-cache versions, `PyTorchMultiHeadAttention`) and plumbing (downloads,
plotting) stay intact, so in `with-headers/` the test suite runs and you only
fill in the concepts the book teaches.

```bash
cd raschka/with-headers
pytest pkg/llms_from_scratch/tests/test_ch04.py -q     # conftest.py puts the package on the path
```
(or `pip install -e .` first.)

**What to implement, by book chapter:**

| Ch | File (`pkg/llms_from_scratch/…`) | You implement | Test |
|---:|----------------------------------|---------------|------|
| 2 | `ch02.py` | `GPTDatasetV1`, `create_dataloader_v1` | `tests/test_ch02.py` |
| 3 | `ch03.py` | `SelfAttention_v1/v2`, `CausalAttention`, `MultiHeadAttention` | `tests/test_ch03.py` |
| 4 | `ch04.py` | `LayerNorm`, `GELU`, `FeedForward`, `TransformerBlock`, `GPTModel`, `generate_text_simple` | `tests/test_ch04.py` |
| 5 | `ch05.py` | `generate`, `train_model_simple`, loss/eval helpers, `load_weights_into_gpt` | `tests/test_ch05.py` |
| 6 | `ch06.py` | `SpamDataset`, `train_classifier_simple`, `classify_review`, … | `tests/test_ch06.py` |
| 7 | `ch07.py` | `format_input`, `InstructionDataset`, `custom_collate_fn` | `tests/test_ch07.py` |

> Chapters 2–4 run fast on CPU with no downloads. Chapter 5's tests can download
> the original OpenAI GPT-2 weights (and want `tensorflow`); chapters 6–7
> download small datasets at runtime. All test files are included — see the
> "Optional / heavy" note in `requirements.txt`. Sample text is in
> `raschka/<tier>/data/the-verdict.txt` so early chapters work offline.

---

## How this was built

`_tools/` holds the (re-runnable) generators:
- `strip_code.py` — AST-based stripper: `strip_source` (keep signatures, drop
  bodies) and `bare_source` (keep only the top comment + imports).
- `nb2py.py` — Jupyter `.ipynb` → `.py` converter (full or `--strip` skeleton).
- `build_exercises.py` — rebuilds both tiers of `karpathy/` and `raschka/` from
  the `*-solutions/` clones. Run `python _tools/build_exercises.py` to regenerate
  (edit the `T_*` strings there to change the top-of-file descriptions).

---

## Verified working

With the bundled `.venv` (Python 3.12, torch 2.12): every generated `.py` file in
both tiers compile-checks clean. In `with-headers/`, the micrograd and Raschka
`test_ch02/03/04` solution tests pass, and the skeletons fail only on the stubbed
functions (`NotImplementedError`). makemore trains end-to-end and minGPT builds a
GPT and runs a forward pass (verified against the solutions).
