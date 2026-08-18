#!/usr/bin/env python3
import json, re, subprocess, time, csv, os
from pathlib import Path

ROOT = Path(os.environ.get("MINIF2F_ROOT", "/tmp/miniF2F"))
SRC = ROOT / "MiniF2F" / "Test.lean"
OUT = Path(os.environ.get("FEB_OUT", "feb_public/results"))
OUT.mkdir(parents=True, exist_ok=True)
MAX_TASKS = int(os.environ.get("MAX_TASKS", "0"))
TIMEOUT = int(os.environ.get("LEAN_TIMEOUT", "25"))

CANDIDATES = [
    "norm_num", "omega", "linarith", "nlinarith", "ring_nf", "aesop", "simp_all", "norm_num at *",
    "omega\n  norm_num at *", "norm_num at *\n  linarith", "norm_num at *\n  nlinarith",
    "ring_nf at *\n  linarith", "ring_nf at *\n  nlinarith", "simp_all\n  norm_num at *", "aesop\n  norm_num at *",
]

def extract_theorems(text):
    pat = re.compile(r"(?ms)^theorem\s+([A-Za-z0-9_']+)\b(.*?):=\s*by\s*\n\s* sorry\s*(?=\n(?:/--|theorem|$))")
    return [(m.group(1), "theorem " + m.group(1) + m.group(2) + ":= by\n  sorry\n") for m in pat.finditer(text)]

def try_proof(decl, tactic):
    proof = "by\n  " + tactic.replace("\n", "\n  ")
    candidate = re.sub(r":=\s*by\s*\n\s*sorry\s*$", ":= " + proof, decl.strip(), flags=re.M)
    f = ROOT / "FEBOne.lean"
    f.write_text("import MiniF2F.ProblemImports\n\n" + candidate + "\n", encoding="utf-8")
    t0=time.time()
    try:
        p=subprocess.run(["lake","env","lean",str(f)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=TIMEOUT)
        return p.returncode==0, time.time()-t0, p.stdout[-4000:]
    except subprocess.TimeoutExpired:
        return False, time.time()-t0, "TIMEOUT"

def main():
    tasks=extract_theorems(SRC.read_text(encoding="utf-8"))
    if MAX_TASKS>0: tasks=tasks[:MAX_TASKS]
    print(f"Extracted {len(tasks)} theorem tasks", flush=True)
    results=[]
    for i,(name,decl) in enumerate(tasks,1):
        solved=False; used=None; trials=0; elapsed=0.0; last=""
        for tactic in CANDIDATES:
            trials+=1; ok,dt,msg=try_proof(decl,tactic); elapsed+=dt; last=msg
            if ok: solved=True; used=tactic; break
        results.append({"index":i,"name":name,"solved":solved,"tactic":used,"trials":trials,"elapsed_s":round(elapsed,3),"last_output":last[-1000:]})
        print(f"[{i}/{len(tasks)}] {name}: {'PASS' if solved else 'FAIL'} trials={trials} t={elapsed:.1f}s", flush=True)
    solved=sum(r["solved"] for r in results)
    summary={"benchmark":"google-deepmind/miniF2F MiniF2F/Test.lean","tasks":len(results),"solved":solved,"pass_rate":solved/len(results) if results else 0,"candidate_budget":len(CANDIDATES),"lean_timeout_s_per_candidate":TIMEOUT,"verification":"lake env lean (Lean kernel)","note":"Frozen deterministic tactic-portfolio public baseline; no theorem-specific tuning during test."}
    (OUT/"minif2f_results.json").write_text(json.dumps({"summary":summary,"results":results},indent=2),encoding="utf-8")
    with (OUT/"minif2f_results.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["index","name","solved","tactic","trials","elapsed_s"]); w.writeheader(); [w.writerow({k:r[k] for k in w.fieldnames}) for r in results]
    print("SUMMARY",json.dumps(summary,sort_keys=True),flush=True)

if __name__ == "__main__": main()
