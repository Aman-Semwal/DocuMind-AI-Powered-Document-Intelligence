"""Start the API and the Streamlit UI with one command (works on Windows, macOS and Linux).

python run_all.py                 # real models via Ollama (settings from .env)
python run_all.py --demo --seed   # offline demo: fake models, sample PDFs uploaded automatically
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def wait_for_service(url: str, process: subprocess.Popen[bytes], timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return False


def seed(base: str, key: str) -> bool:
    pdf_dir = ROOT / "sample_pdfs"
    for pdf in sorted(pdf_dir.glob("Northwind_*.pdf")):
        with pdf.open("rb") as fh:
            r = httpx.post(
                f"{base}/documents/upload",
                headers={"X-API-Key": key},
                files={"file": (pdf.name, fh, "application/pdf")},
                timeout=60,
            )
        print(f"  seed {pdf.name}: HTTP {r.status_code}")
        if r.status_code not in (202, 409):
            print(f"    {r.text}")
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="offline demo with fake models (no Ollama needed)")
    ap.add_argument("--seed", action="store_true", help="upload the sample PDFs once the API is up")
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--ui-port", type=int, default=8501)
    args = ap.parse_args()

    env = os.environ.copy()
    env.update(load_env(ROOT / (".env.demo" if args.demo else ".env")))
    if not args.demo and not (ROOT / ".env").exists():
        print("No .env found. Copy .env.example to .env first (cp .env.example .env).")
        return 1
    key = env.get("API_KEY") or (env.get("API_KEYS", "").split(",")[0].strip())
    if not key:
        print("Set API_KEYS (and API_KEY) in .env.")
        return 1
    base = f"http://localhost:{args.api_port}"
    env.update({"API_KEY": key, "API_BASE_URL": base, "PYTHONPATH": str(ROOT)})

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:get_app", "--factory", "--port", str(args.api_port)],
        cwd=ROOT,
        env=env,
        creationflags=creationflags,
    )
    procs = [api]
    try:
        print(f"Starting API on {base} ...")
        if not wait_for_service(f"{base}/health", api):
            print("The API did not start. See the log above.")
            return 1
        if args.seed and not seed(base, key):
            print("Sample document seeding failed.")
            return 1
        ui = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "frontend/app.py",
                "--server.port",
                str(args.ui_port),
                "--browser.gatherUsageStats",
                "false",
            ],
            cwd=ROOT,
            env=env,
            creationflags=creationflags,
        )
        procs.append(ui)
        if not wait_for_service(f"http://localhost:{args.ui_port}/_stcore/health", ui):
            print("The UI did not start. See the log above.")
            return 1
        print(
            f"\n  UI:      http://localhost:{args.ui_port}\n  Swagger: {base}/docs   (API key: {key})\n  Ctrl+C stops both.\n"
        )
        while all(p.poll() is None for p in procs):
            time.sleep(0.5)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    sys.exit(main())
