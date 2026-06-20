#!/usr/bin/env python3
"""
strip_code.py — turn a finished Python module into an exercise skeleton.

For every targeted function / method it keeps:
  * imports, module-level constants, decorators
  * the full signature (including multi-line signatures and annotations)
  * the docstring (if any)
  * any *leading* comment lines inside the body (kept as hints)
and replaces the rest of the body with:

      # TODO: implement <qualname>
      raise NotImplementedError

The transformation is done with byte-accurate source surgery driven by the
AST, so everything outside the targeted bodies (comments, blank lines, exact
formatting) is preserved verbatim.

Usage:
    python strip_code.py FILE [--names A,B,C] [--out OUT | --inplace]
    python strip_code.py --selftest

--names limits stripping to the given top-level classes/functions (a class
name strips all of its methods). With no --names, every top-level function and
every method of every class is stripped.
"""
from __future__ import annotations
import ast
import re
import sys
import argparse


def _line_starts(b: bytes) -> list[int]:
    starts = [0]
    for i, ch in enumerate(b):
        if ch == 0x0A:  # '\n'
            starts.append(i + 1)
    return starts


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(getattr(node, "value", None), ast.Constant)
        and isinstance(node.value.value, str)
    )


def strip_source(source: str, target_names: set[str] | None = None,
                 hint: bool = True) -> str:
    b = source.encode("utf-8")
    tree = ast.parse(source)
    line_starts = _line_starts(b)

    def off(lineno: int, col: int) -> int:
        return line_starts[lineno - 1] + col

    # parent map
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def enclosing_class(fn: ast.AST) -> str | None:
        p = parent.get(fn)
        while p is not None:
            if isinstance(p, ast.ClassDef):
                return p.name
            if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return None
            p = parent.get(p)
        return None

    def nested_in_function(fn: ast.AST) -> bool:
        p = parent.get(fn)
        while p is not None:
            if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return True
            p = parent.get(p)
        return False

    def is_target(fn: ast.AST) -> bool:
        if nested_in_function(fn):
            return False  # outer body replacement subsumes nested defs
        if target_names is None:
            return True
        cls = enclosing_class(fn)
        return (cls in target_names) if cls else (fn.name in target_names)

    targets = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_target(n)
    ]

    repls: list[tuple[int, int, bytes]] = []
    for fn in targets:
        body = fn.body
        doc = body[0] if (body and _is_docstring(body[0])) else None
        rest = body[1:] if doc is not None else body

        cls = enclosing_class(fn)
        qual = f"{cls}.{fn.name}" if cls else fn.name
        comment = f"# TODO: implement {qual}" if hint else "# TODO: implement"

        if rest:
            anchor = rest[0]
            last = rest[-1]
            start_b = off(anchor.lineno, anchor.col_offset)
            end_b = off(last.end_lineno, last.end_col_offset)
            # drop a trailing same-line comment (it now refers to removed code)
            line_end = b.find(b"\n", end_b)
            if line_end == -1:
                line_end = len(b)
            trailing = b[end_b:line_end]
            if trailing.strip() == b"" or trailing.lstrip().startswith(b"#"):
                end_b = line_end
            prefix = b[line_starts[anchor.lineno - 1]:start_b]
            if prefix.strip() == b"":
                # body statement starts on its own line; reuse its indentation
                indent = " " * anchor.col_offset
                text = comment + "\n" + indent + "raise NotImplementedError"
            else:
                # one-liner: `def f(): return x`
                indent = " " * (fn.col_offset + 4)
                text = "\n" + indent + comment + "\n" + indent + "raise NotImplementedError"
            repls.append((start_b, end_b, text.encode("utf-8")))
        elif doc is not None:
            # docstring-only body: append the stub after the docstring
            ins_b = off(doc.end_lineno, doc.end_col_offset)
            indent = " " * (fn.col_offset + 4)
            text = "\n" + indent + comment + "\n" + indent + "raise NotImplementedError"
            repls.append((ins_b, ins_b, text.encode("utf-8")))

    repls.sort(key=lambda r: r[0], reverse=True)
    out = b
    for start_b, end_b, txt in repls:
        out = out[:start_b] + txt + out[end_b:]
    result = out.decode("utf-8")
    # sanity: the skeleton must still parse
    ast.parse(result)
    return result


def extract_import_segments(source: str) -> list[str]:
    """Return the source text of every top-level import statement, in order."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # line-based fallback (e.g. notebook conversions that don't fully parse)
        out = []
        for line in source.splitlines():
            if re.match(r"(import\s|from\s+\S+\s+import\b)", line):
                out.append(line.rstrip())
        return out
    b = source.encode("utf-8")
    starts = _line_starts(b)
    segs = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            s = starts[node.lineno - 1] + node.col_offset
            e = starts[node.end_lineno - 1] + node.end_col_offset
            segs.append(b[s:e].decode("utf-8"))
    return segs


def bare_source(source: str, topline: str = "") -> str:
    """Reduce a module to just a top-level comment and its imports."""
    segs = extract_import_segments(source)
    out = topline.rstrip() + "\n" if topline else ""
    if segs:
        out += "\n" + "\n".join(segs) + "\n"
    return out.lstrip("\n") if out else "\n"


# --------------------------------------------------------------------------- #
_SELFTEST_SRC = '''\
"""module doc."""
import math
CONST = 42

def add(a, b):
    """Return a+b."""
    # this is a hint
    s = a + b
    return s

def oneliner(x): return x * 2

class Foo:
    """class doc."""
    attr = 1
    def __init__(self, n):
        self.n = n
        self.vals = [i for i in range(n)]
    def method(self,
               y):
        return self.n + y
    @staticmethod
    def stat():
        return 99
    def nester(self):
        def inner():
            return 1
        return inner()
'''


def _selftest() -> None:
    out = strip_source(_SELFTEST_SRC)
    print(out)
    print("=" * 60)
    compile(out, "<skeleton>", "exec")
    assert "import math" in out
    assert "CONST = 42" in out
    assert "attr = 1" in out
    assert out.count("raise NotImplementedError") == 6  # add, oneliner, __init__, method, stat, nester (inner subsumed)
    assert "# this is a hint" in out  # leading comment preserved
    assert "def inner" not in out      # nested def stripped away
    # selective stripping
    out2 = strip_source(_SELFTEST_SRC, target_names={"add"})
    assert out2.count("raise NotImplementedError") == 1
    assert "return self.n + y" in out2  # Foo.method untouched
    # bare mode: only the topline comment + imports survive
    bare = bare_source(_SELFTEST_SRC, "# top comment")
    print("--- bare ---")
    print(bare)
    assert bare.startswith("# top comment")
    assert "import math" in bare
    assert "def add" not in bare and "class Foo" not in bare and "CONST" not in bare
    print("SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?")
    ap.add_argument("--names", default=None, help="comma-separated class/func names")
    ap.add_argument("--out", default=None)
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.file:
        ap.error("file required")

    with open(args.file, encoding="utf-8") as f:
        src = f.read()
    names = set(args.names.split(",")) if args.names else None
    out = strip_source(src, names)
    if args.inplace:
        with open(args.file, "w", encoding="utf-8") as f:
            f.write(out)
    elif args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    main()
