"""Execute local setup/import prefixes in fresh ARENA kernels and record exact coverage.

This does not fill exercise blanks, run paid APIs, or claim whole-paper replication.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient

from launch import configure_environment

ROOT = Path(__file__).resolve().parent.parent
ARENA = ROOT / "arena"

READ_ONLY_NETWORK = r'''
import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["MPLBACKEND"] = "Agg"
import requests, httpx
from urllib.parse import urlparse
_paid_hosts = {"api.openai.com", "api.anthropic.com", "openrouter.ai", "api.together.xyz", "api.deepinfra.com"}
def _check_network(method, url):
    host = urlparse(str(url)).hostname or ""
    if host in _paid_hosts or "ndif" in host:
        raise RuntimeError("MANUAL_API: live model API validation is disabled by the learner's instruction")
    if method.upper() not in {"GET", "HEAD", "OPTIONS"} and host not in {"localhost", "127.0.0.1"}:
        if not (host == "huggingface.co" and str(url).endswith("paths-info")):
            raise RuntimeError("MANUAL_NETWORK_WRITE: external writes are disabled during validation")
_requests_request = requests.Session.request
def _safe_request(self, method, url, *args, **kwargs):
    _check_network(method, url)
    return _requests_request(self, method, url, *args, **kwargs)
requests.Session.request = _safe_request
_httpx_send = httpx.Client.send
def _safe_send(self, request, *args, **kwargs):
    _check_network(request.method, request.url)
    return _httpx_send(self, request, *args, **kwargs)
httpx.Client.send = _safe_send
_httpx_async_send = httpx.AsyncClient.send
async def _safe_async_send(self, request, *args, **kwargs):
    _check_network(request.method, request.url)
    return await _httpx_async_send(self, request, *args, **kwargs)
httpx.AsyncClient.send = _safe_async_send
'''


def validate(record):
    path = ROOT / record["local"]
    original = nbformat.read(path, as_version=4)
    nbformat.validate(original)
    code_cells = [(i, c) for i, c in enumerate(original.cells) if c.cell_type == "code"]
    # The replaced bootstrap and original import/configuration cell establish local startup.
    prefix = code_cells[:2]
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(READ_ONLY_NETWORK)] + [c for _, c in prefix], metadata=original.metadata)
    started = time.monotonic()
    result = {"local": record["local"], "kind": record["kind"], "coverage": "local setup and original import/configuration cell",
              "original_cells": [i + 1 for i, _ in prefix], "status": "passed", "error": None}
    try:
        NotebookClient(notebook, timeout=180, kernel_name=notebook.metadata.kernelspec.name,
                       resources={"metadata": {"path": str(path.parent)}}).execute()
    except Exception as error:
        text = str(error)
        if any(key in text for key in ["MANUAL_API", "API key", "API_KEY", "HF_TOKEN", "gated repo", "GatedRepoError"]):
            result["status"] = "manual_credentials"
        else:
            result["status"] = "failed"
        result["error"] = text[-5500:]
    result["seconds"] = round(time.monotonic() - started, 2)
    output_path = ARENA / "validation" / "executed" / Path(record["local"]).relative_to("arena")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", default="")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    configure_environment()
    manifest = json.loads((ARENA / "manifest.json").read_text(encoding="utf-8"))
    records = [r for r in manifest["notebooks"] if args.match in r["local"]]
    output = ARENA / "validation" / "setup-results.json"
    output.parent.mkdir(exist_ok=True)
    old = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"results": []}
    merged = {r["local"]: r for r in old["results"]}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(validate, r): r for r in records}
        for future in as_completed(futures):
            result = future.result()
            merged[result["local"]] = result
            output.write_text(json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(), "results": list(merged.values())}, indent=2), encoding="utf-8")
            print(f'{result["status"]:18} {Path(result["local"]).name}', flush=True)
    counts = {status: sum(r["status"] == status for r in merged.values()) for status in {r["status"] for r in merged.values()}}
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
