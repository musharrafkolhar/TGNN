"""Launch the next-generation TGNN Fraud Intelligence Platform."""

import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

PORT = 8000
HOST = "127.0.0.1"
URL = f"http://{HOST}:{PORT}"


def open_browser():
    time.sleep(1.2)
    print(f"\nOpening TGNN Fraud Intelligence Platform at {URL} ...\n")
    webbrowser.open(URL)


if __name__ == "__main__":
    # Ensure current directory is on sys.path
    root_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(root_dir / "src"))

    threading.Thread(target=open_browser, daemon=True).start()
    print("=" * 70)
    print("  TGNN FRAUD INTELLIGENCE PLATFORM (ENTERPRISE AML SUITE)")
    print(f"  Live Server running on {URL}")
    print("  Press Ctrl+C to terminate the server.")
    print("=" * 70)
    uvicorn.run("app.server:app", host=HOST, port=PORT, log_level="info")
