"""Launch the TGNN Fraud Intelligence Platform with automatic port detection and browser launch."""

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn


def find_available_port(start_port: int = 8000, max_attempts: int = 20) -> int:
    """Find a free TCP port to prevent bind errors."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port


def launch_browser(url: str):
    """Reliably open browser on Windows using startfile or webbrowser."""
    time.sleep(1.0)
    print(f"\n>>> Launching browser at: {url}\n")
    try:
        if sys.platform == "win32":
            os.startfile(url)
        else:
            webbrowser.open(url)
    except Exception:
        webbrowser.open(url)


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(root_dir / "src"))

    # Determine port
    port = find_available_port(8000)
    url = f"http://127.0.0.1:{port}"

    print("=" * 72)
    print("  TGNN FRAUD INTELLIGENCE PLATFORM (ENTERPRISE AML SUITE)")
    print(f"  Live Server: {url}")
    print("  Status: All genuine offline runs, metrics, and alerts ready.")
    print("=" * 72)

    # Launch browser in background thread
    threading.Thread(target=launch_browser, args=(url,), daemon=True).start()

    # Start Uvicorn
    try:
        uvicorn.run("app.server:app", host="127.0.0.1", port=port, log_level="info")
    except Exception as e:
        print(f"\nServer error: {e}")
        input("\nPress Enter to exit...")
