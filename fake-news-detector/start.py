"""
🚀 Fake News Detector — One-Click Launcher

Starts both the API server and Streamlit dashboard in a single command.

Usage:
    python start.py          → Start API + Streamlit
    python start.py --api    → Start API server only
    python start.py --app    → Start Streamlit app only
"""

import os
import sys
import signal
import subprocess
import time
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ANSI colors for terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

processes = []


def log(tag, color, msg):
    print(f"{color}{BOLD}[{tag}]{RESET} {msg}")


def check_model():
    """Check if the trained model exists."""
    model_path = os.path.join(PROJECT_ROOT, "model", "fake_news_model.pkl")
    if not os.path.exists(model_path):
        log("WARNING", YELLOW,
            "No trained model found at model/fake_news_model.pkl")
        log("WARNING", YELLOW,
            "Run 'python scripts/train_model.py' first to train the model.")
        log("WARNING", YELLOW,
            "Starting anyway — the app will work in limited mode.\n")
        return False
    log("MODEL", GREEN, "Trained model found ✓")
    return True


def start_api():
    """Start the FastAPI server."""
    log("API", CYAN, "Starting FastAPI server on http://localhost:8000 ...")
    return subprocess.Popen(
        [sys.executable, os.path.join("src", "api_server.py")],
        cwd=PROJECT_ROOT,
    )


def start_streamlit():
    """Start the Streamlit dashboard."""
    log("APP", CYAN, "Starting Streamlit dashboard on http://localhost:8501 ...")
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run",
         os.path.join("src", "app.py"),
         "--server.headless", "true"],
        cwd=PROJECT_ROOT,
    )


def shutdown(sig=None, frame=None):
    """Gracefully shut down all child processes."""
    print()
    log("SHUTDOWN", YELLOW, "Stopping all services...")
    for proc in processes:
        try:
            proc.terminate()
        except Exception:
            pass
    # Give them a moment to exit cleanly
    for proc in processes:
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    log("SHUTDOWN", GREEN, "All services stopped. Goodbye! 👋")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Fake News Detector Launcher")
    parser.add_argument("--api", action="store_true",
                        help="Start API server only")
    parser.add_argument("--app", action="store_true",
                        help="Start Streamlit app only")
    args = parser.parse_args()

    # If neither flag is set, start both
    start_both = not args.api and not args.app

    print()
    print(f"{CYAN}{BOLD}╔══════════════════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}{BOLD}║       🛡️  Fake News Detector — Launcher              ║{RESET}")
    print(f"{CYAN}{BOLD}╚══════════════════════════════════════════════════════╝{RESET}")
    print()

    check_model()

    # Register Ctrl+C handler
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if start_both or args.api:
        processes.append(start_api())
        time.sleep(1)  # Let API server grab its port first

    if start_both or args.app:
        processes.append(start_streamlit())

    print()
    log("READY", GREEN, "All services are running! 🎉")
    print()
    if start_both or args.api:
        print(f"   🌐 API Server:    {BOLD}http://localhost:8000{RESET}")
        print(f"   📖 API Docs:      {BOLD}http://localhost:8000/docs{RESET}")
    if start_both or args.app:
        print(f"   🖥️  Dashboard:     {BOLD}http://localhost:8501{RESET}")
    print()
    log("INFO", YELLOW, "Press Ctrl+C to stop all services.\n")

    # Wait for any process to exit
    try:
        while True:
            for proc in processes:
                retcode = proc.poll()
                if retcode is not None:
                    name = "API Server" if proc == processes[0] else "Streamlit"
                    log("EXIT", RED,
                        f"{name} exited with code {retcode}")
                    shutdown()
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
