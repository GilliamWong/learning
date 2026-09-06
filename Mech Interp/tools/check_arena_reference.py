"""Run upstream numerical tests against upstream reference functions in isolated processes."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib
import inspect
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
ARENA = ROOT / "arena"


def worker(chapter, section):
    result = {"chapter": chapter, "section": section, "passed": [], "failed": [], "needs_fixture": [], "deferred": [],
              "coverage": "available numerical reference tests; no full training or paid API calls", "status": "passed"}
    output = ARENA / "validation" / "reference" / f"{chapter}--{section}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        runtime = runpy.run_path(str(ARENA / "runtime.py"))
        context = runtime["setup"](chapter, section)
        from validate_arena import READ_ONLY_NETWORK
        exec(READ_ONLY_NETWORK, {})
        import plotly.basedatatypes
        plotly.basedatatypes.BaseFigure.show = lambda self, *args, **kwargs: None
        import matplotlib.pyplot as plt
        plt.show = lambda *args, **kwargs: None
        if section == "part4_rlhf":
            # Match the notebook's import order around the upstream circular import.
            importlib.import_module(f"{section}.tests_lora")
        module_names = ["solutions_vaes", "solutions_gans"] if section == "part5_vaes_and_gans" else ["solutions"]
        references = [importlib.import_module(f"{section}.{name}") for name in module_names]
        tests = importlib.import_module(f"{section}.tests")
        namespace = {}
        for reference in references:
            namespace.update(vars(reference))
        if section == "part5_mcts_alphazero":
            namespace["model"] = namespace["Connect4Model"](namespace["device"]).eval()
            namespace["batched"] = namespace["BatchedMCTS"](namespace["Connect4Env"](device=namespace["device"]),
                                                           namespace["MCTSConfig"](sims=64, c_puct=1.5))
        source = "\n".join(Path(reference.__file__).read_text(encoding="utf-8") for reference in references)
        tree = ast.parse(source)
        calls = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "tests" and node.func.attr.startswith("test_"):
                    calls.setdefault(ast.unparse(node), node)
        invoked = set()
        for label, call in calls.items():
            name = call.func.attr
            if ((section == "part3_optimization" and name in {"test_broadcast", "test_reduce", "test_all_reduce"})
                or (section == "part3_ppo" and any(mode in label for mode in ["mode='atari'", "mode='mujoco'"]))):
                result["deferred"].append({"test": label, "reason": "Learner deferred Linux-only or multi-GPU extensions."})
                continue
            try:
                # Use precisely the reference arguments specified by the teaching code.
                args = [eval(compile(ast.Expression(arg), "<reference argument>", "eval"), namespace) for arg in call.args]
                kwargs = {kw.arg: eval(compile(ast.Expression(kw.value), "<reference argument>", "eval"), namespace) for kw in call.keywords if kw.arg}
            except (NameError, AttributeError) as error:
                result["needs_fixture"].append({"test": label, "reason": str(error)})
                continue
            if any("client" in ast.unparse(arg).lower() for arg in call.args):
                result["needs_fixture"].append({"test": label, "reason": "Live API test remains manual."})
                continue
            try:
                getattr(tests, name)(*args, **kwargs)
                result["passed"].append(label)
                invoked.add(name)
            except NameError as error:
                result["needs_fixture"].append({"test": label, "reason": f"Sequential notebook state required: {error}"})
            except Exception as error:
                result["failed"].append({"test": label, "error": f"{type(error).__name__}: {error}"})
        # Some simple reference tests are omitted from the demo's explicit calls.
        for name, function in vars(tests).items():
            if not (name.startswith("test_") and inspect.isfunction(function)) or name in invoked or any(c.func.attr == name for c in calls.values()):
                continue
            args = []
            missing = []
            for param in inspect.signature(function).parameters.values():
                if param.default is not inspect.Parameter.empty:
                    continue
                candidates = [param.name, param.name.removeprefix("my_").removesuffix("Class")]
                if param.name in {"fn", "test_fn"}:
                    candidates.append(name.removeprefix("test_"))
                match = next((candidate for candidate in candidates if candidate in namespace), None)
                if match and "client" not in param.name.lower():
                    args.append(namespace[match])
                else:
                    missing.append(param.name)
            if missing:
                result["needs_fixture"].append({"test": name, "reason": "Requires explicit fixture(s): " + ", ".join(missing)})
                continue
            try:
                function(*args)
                result["passed"].append(name)
            except Exception as error:
                result["failed"].append({"test": name, "error": f"{type(error).__name__}: {error}"})
        result["status"] = "failed" if result["failed"] else "partial" if result["needs_fixture"] or result["deferred"] else "passed" if result["passed"] else "needs_fixture"
    except Exception as error:
        result["status"] = "manual_credentials" if any(key in str(error) for key in ["API_KEY", "API key", "HF_TOKEN", "MANUAL_API", "gated repo"]) else "needs_fixture" if isinstance(error, (NameError, ModuleNotFoundError)) else "failed"
        result["import_error"] = traceback.format_exc()[-5000:]
    result["seconds"] = round(time.monotonic() - started, 2)
    result["checked_at"] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"section": section, "status": result["status"], "passed": len(result["passed"]), "failed": len(result["failed"])}), flush=True)


def run_section(chapter, section):
    log_dir = ARENA / "validation" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = ARENA / "circuits" if section == "part42_sae_circuits" else ARENA
    command = [str(environment / ".venv" / "Scripts" / "python.exe"), "-X", "utf8", str(Path(__file__).resolve()), "--worker", chapter, section]
    log = log_dir / f"reference--{chapter}--{section}.log"
    with log.open("w", encoding="utf-8") as stream:
        try:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=stream, timeout=360, check=True)
        except subprocess.TimeoutExpired:
            record = {"chapter": chapter, "section": section, "status": "not_completed", "reason": "Reference checks exceeded the six-minute validation bound; no success is claimed."}
            output = ARENA / "validation" / "reference" / f"{chapter}--{section}.json"
            output.parent.mkdir(exist_ok=True)
            output.write_text(json.dumps(record, indent=2), encoding="utf-8")
        except subprocess.CalledProcessError as error:
            return {"chapter": chapter, "section": section, "status": "failed", "reason": str(error), "log": str(log)}
    output = ARENA / "validation" / "reference" / f"{chapter}--{section}.json"
    return json.loads(output.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--match", default="")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
        return
    manifest = json.loads((ARENA / "manifest.json").read_text(encoding="utf-8"))
    sections = []
    for record in manifest["notebooks"]:
        if record["kind"] == "exercise" and any(term in record["local"] for term in args.match.split(",")):
            parts = Path(record["source"]).parts
            sections.append((parts[0], parts[2]))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(run_section, *section) for section in sections]):
            result = future.result()
            print(f'{result["status"]:18} {result["section"]}: {len(result.get("passed", []))} tests passed, {len(result.get("failed", []))} failed', flush=True)


if __name__ == "__main__":
    main()
