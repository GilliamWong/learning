"""Small, documented compatibility repairs to upstream reference support code."""

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "arena" / "ARENA_3.0"


def reference_repairs(text, topic):
    if topic == "mcts":
        text = text.replace("(B, 1) sampled column indices (one per game).", "(B,) sampled column indices (one per game).")
        text = re.sub(r"(action = torch\.multinomial\(probs, num_samples=1\)\n        return action)(?:\.squeeze\(-1\))*$",
                      r"\1.squeeze(-1)", text, flags=re.M)
    if topic == "optimization":
        text = text.replace("tests.test_all_reduce(ring_all_reduce)", "tests.test_all_reduce(ring_all_reduce, WORLD_SIZE)")
    if topic == "control":
        text = text.replace('local_exec = "python3"', 'local_exec = sys.executable')
        text = text.replace('    await sandbox().exec(["mkdir", "-p", "/tmp/sandbox"])\n    await sandbox().write_file("/tmp/sandbox/code.py", code)',
                            '    sandbox_dir = "/tmp/sandbox" if USING_DOCKER else "."\n    if USING_DOCKER:\n        await sandbox().exec(["mkdir", "-p", sandbox_dir])\n    await sandbox().write_file(f"{sandbox_dir}/code.py", code)')
        text = text.replace('sys.path.insert(0, "/tmp/sandbox")', 'sys.path.insert(0, {sandbox_dir!r})')
        text = text.replace('await sandbox().write_file("/tmp/sandbox/runner.py", runner_script)', 'await sandbox().write_file(f"{sandbox_dir}/runner.py", runner_script)')
        text = text.replace('sandbox().exec([py, "/tmp/sandbox/runner.py"])', 'sandbox().exec([py, f"{sandbox_dir}/runner.py"])')
    return text


def apply():
    records = []
    items = [
        ("chapter0_fundamentals/exercises/part3_optimization/solutions.py", "optimization", "Supply the required world_size argument in the optional ring-all-reduce test call; distributed execution remains deferred."),
        ("chapter2_rl/exercises/part5_mcts_alphazero/solutions.py", "mcts", "Align sampled-action shape with Connect4Env and the provided test: (batch,)."),
        ("chapter3_llm_evals/exercises/part5_ai_control/solutions.py", "control", "Use the active Python and relative sandbox files in the explicit local test branch; keep Docker as the default."),
    ]
    for relative, topic, reason in items:
        path = SOURCE / relative
        original = path.read_text(encoding="utf-8")
        updated = reference_repairs(original, topic)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
        records.append({"file": relative, "reason": reason, "current_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (ROOT / "arena" / "support-patches.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


if __name__ == "__main__":
    apply()
