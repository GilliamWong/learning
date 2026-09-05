"""Start/reuse/stop this project's local JupyterLab; no global configuration changes."""

import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen
import webbrowser

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / ".runtime"
CONNECTION = RUNTIME / "connection.json"


def read_connection():
    try:
        info = json.loads(CONNECTION.read_text(encoding="utf-8"))
        if info["root"] != str(ROOT) or not 1 <= int(info["port"]) <= 65535:
            return None
        return info
    except (OSError, ValueError, KeyError, TypeError):
        return None


def request(info, endpoint="/api", method="GET"):
    url = f'http://127.0.0.1:{info["port"]}{endpoint}'
    req = Request(url, method=method, headers={"Authorization": f'token {info["token"]}'},
                  data=b"" if method == "POST" else None)
    with urlopen(req, timeout=2) as response:
        return response.status


def ready(info):
    try:
        return bool(info and request(info) == 200)
    except (OSError, ValueError):
        return False


def configure_environment():
    for name, relative in {
        "JUPYTER_CONFIG_DIR": "jupyter/config", "JUPYTER_DATA_DIR": "jupyter/data",
        "JUPYTER_RUNTIME_DIR": "jupyter/runtime", "IPYTHONDIR": "ipython", "MPLCONFIGDIR": "matplotlib",
    }.items():
        path = RUNTIME / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["PYTHONUTF8"] = "1"


def serve():
    configure_environment()
    info = read_connection()
    from jupyterlab.labapp import LabApp
    LabApp.launch_instance(argv=[
        "--no-browser", "--ip=127.0.0.1", f'--port={info["port"]}', "--ServerApp.port_retries=0",
        f"--ServerApp.root_dir={ROOT}", f'--IdentityProvider.token={info["token"]}',
        "--LabApp.default_url=/lab/tree/00_Start_Here.ipynb",
        "--ServerApp.allow_remote_access=False",
    ])


def start(open_browser=True):
    from filelock import FileLock
    RUNTIME.mkdir(exist_ok=True)
    with FileLock(str(RUNTIME / "launch.lock"), timeout=45):
        info = read_connection()
        if not ready(info):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            token = secrets.token_urlsafe(32)
            info = {"root": str(ROOT), "port": port, "token": token,
                    "url": f"http://127.0.0.1:{port}/lab/tree/00_Start_Here.ipynb?token={token}"}
            CONNECTION.write_text(json.dumps(info), encoding="utf-8")
            configure_environment()
            with (RUNTIME / "server.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "serve"],
                                           cwd=ROOT, stdout=log, stderr=log,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            info["pid"] = process.pid
            CONNECTION.write_text(json.dumps(info), encoding="utf-8")
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                if ready(info):
                    break
                if process.poll() is not None:
                    raise RuntimeError(f"Jupyter stopped while starting. See {RUNTIME / 'server.log'}")
                time.sleep(0.25)
            else:
                raise RuntimeError(f"Jupyter did not become ready. See {RUNTIME / 'server.log'}")
    if open_browser:
        webbrowser.open(info["url"])
    print("Workbench ready. Click Open / refresh workbench on the home notebook.")


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "start"
    if command == "serve":
        serve()
    elif command == "start":
        start("--no-browser" not in sys.argv)
    elif command == "status":
        info = read_connection()
        print(json.dumps({"running": ready(info), "url": info["url"] if info else None, "pid": info.get("pid") if info else None}))
    elif command == "stop":
        info = read_connection()
        if ready(info):
            request(info, "/api/shutdown", "POST")
            print("Workbench stopped. Your saved files and progress remain in this folder.")
        else:
            print("This workbench is not running.")
    else:
        raise SystemExit("Use start, status, or stop.")


if __name__ == "__main__":
    main()
