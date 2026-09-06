"""Fetch the public supporting repositories explicitly referenced by ARENA."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "arena" / "ARENA_3.0"
LOCK_PATH = ROOT / "arena" / "supporting-repositories.json"
LOCKED = {r["url"]: r for r in json.loads(LOCK_PATH.read_text(encoding="utf-8"))["repositories"] if "url" in r} if LOCK_PATH.exists() else {}
REPOSITORIES = [
    ("saprmarks/geometry-of-truth", "chapter1_transformer_interp/exercises/geometry-of-truth"),
    ("ApolloResearch/deception-detection", "chapter1_transformer_interp/exercises/deception-detection"),
    ("decoderesearch/circuit-tracer", "chapter1_transformer_interp/exercises/circuit-tracer"),
    ("neelnanda-io/Grokking", "chapter1_transformer_interp/exercises/part52_grokking_and_modular_arithmetic/Grokking"),
    ("clarifying-EM/model-organisms-for-EM", "chapter4_alignment_science/exercises/model-organisms-for-EM"),
    ("PalisadeResearch/shutdown_avoidance", "chapter4_alignment_science/exercises/shutdown_avoidance"),
    ("interp-reasoning/thought-anchors", "chapter4_alignment_science/exercises/thought-anchors"),
    ("safety-research/assistant-axis", "chapter4_alignment_science/exercises/assistant-axis"),
    ("tim-hua-01/ai-psychosis", "chapter4_alignment_science/exercises/ai-psychosis"),
    ("safety-research/petri", "chapter4_alignment_science/exercises/petri"),
]


def fetch(item):
    repository, relative = item
    target = (SOURCE / relative).resolve()
    assert target.is_relative_to(SOURCE.resolve())
    url = f"https://github.com/{repository}.git"
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1")
    pinned = LOCKED.get(url, {})
    if not target.exists():
        subprocess.run(["git", "-c", "core.longpaths=true", "clone", "--no-checkout", "--depth", "1", url, str(target)],
                       env=env, check=True, capture_output=True, text=True)
        revision = pinned.get("revision", "origin/HEAD")
        if pinned.get("revision"):
            subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", revision], env=env, check=True, capture_output=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", revision], env=env, check=True, capture_output=True)
    def git(*args):
        return subprocess.check_output(["git", "-c", f"safe.directory={target}", "-C", str(target), *args], text=True).strip()
    assert git("remote", "get-url", "origin") == url
    revision = git("rev-parse", "HEAD")
    if pinned.get("revision") and revision != pinned["revision"]:
        raise RuntimeError(f"The supporting checkout has changed: {target}. Review it before updating the recorded pin.")
    return {**pinned, "url": url, "path": target.relative_to(ROOT).as_posix(), "revision": revision,
            "lfs": "Large Git-LFS artifacts are downloaded only when a specific exercise needs them."}


if __name__ == "__main__":
    if not SOURCE.exists():
        manifest = json.loads((ROOT / "arena/manifest.json").read_text(encoding="utf-8"))
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1")
        subprocess.run(["git", "-c", "core.longpaths=true", "clone", "--no-checkout", "--depth", "1", manifest["upstream"], str(SOURCE)], env=env, check=True)
        subprocess.run(["git", "-C", str(SOURCE), "fetch", "--depth", "1", "origin", manifest["revision"]], env=env, check=True)
        subprocess.run(["git", "-C", str(SOURCE), "checkout", "--detach", manifest["revision"]], env=env, check=True)
    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(fetch, r) for r in REPOSITORIES]):
            try:
                result = future.result()
                records.append(result)
                print("Fetched", result["url"], flush=True)
            except Exception as error:
                records.append({"error": str(error)})
                print("Fetch failed:", error, flush=True)
    if any("error" in r for r in records):
        raise SystemExit("A supporting repository could not be prepared; the previous pin record was preserved.")
    output = {"fetched_at": datetime.now(timezone.utc).isoformat(), "repositories": records}
    LOCK_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    from patch_arena_support import apply
    apply()
