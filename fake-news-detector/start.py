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
import io

# Fix Unicode output on Windows consoles (cp1252 can't print box-drawing/emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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
    """Check if the trained model exists, and train it automatically if missing."""
    model_path = os.path.join(PROJECT_ROOT, "model", "fake_news_model.pkl")
    if not os.path.exists(model_path):
        log("TRAINING", YELLOW, "No trained model found. Automatically training ensemble model...")
        py_exec = get_python_executable()
        try:
            # Run the training script synchronously and wait for it to complete
            subprocess.run([py_exec, os.path.join("scripts", "train_model.py")], check=True)
            log("TRAINING", GREEN, "Model training completed successfully! ✓")
            return True
        except Exception as e:
            log("ERROR", RED, f"Failed to automatically train model: {e}")
            log("WARNING", YELLOW, "Starting anyway — the app will work in limited mode.\n")
            return False
    log("MODEL", GREEN, "Trained model found ✓")
    return True


def get_python_executable():
    """Resolve the python executable to use. Prefer the virtual environment's python."""
    # Check for Windows virtualenv python
    venv_py_win = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_py_win):
        return venv_py_win
    # Check for Unix virtualenv python
    venv_py_unix = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    if os.path.exists(venv_py_unix):
        return venv_py_unix
    # Fallback to sys.executable
    return sys.executable


def free_port(port):
    """Kill any process occupying the given port (Windows-only)."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('127.0.0.1', port))
            if result != 0:
                return  # Port is free
    except Exception:
        return
    # Port is in use — try to free it
    log("PORT", YELLOW, f"Port {port} is in use. Attempting to free it...")
    try:
        out = subprocess.check_output(
            f'netstat -ano | findstr ":{port}"',
            shell=True, text=True, stderr=subprocess.DEVNULL
        )
        pids = set()
        for line in out.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 5 and f':{port}' in parts[1]:
                pid = parts[-1]
                if pid.isdigit() and int(pid) != os.getpid():
                    pids.add(int(pid))
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                log("PORT", GREEN, f"Killed PID {pid} holding port {port}")
            except (PermissionError, OSError):
                pass
        if pids:
            time.sleep(1)
    except Exception:
        pass


def start_api():
    """Start the FastAPI server."""
    log("API", CYAN, "Starting FastAPI server on http://localhost:8000 ...")
    py_exec = get_python_executable()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(
        [py_exec, os.path.join("src", "api_server.py")],
        cwd=PROJECT_ROOT,
        env=env,
    )


def start_updater():
    """Start the real-time background update daemon."""
    log("UPDATE", CYAN, "Starting real-time background update daemon (30 min cycle)...")
    py_exec = get_python_executable()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(
        [py_exec, os.path.join("scripts", "realtime_update.py")],
        cwd=PROJECT_ROOT,
        env=env,
    )


def start_streamlit():
    """Start the Streamlit dashboard."""
    log("APP", CYAN, "Starting Streamlit dashboard on http://localhost:8501 ...")
    py_exec = get_python_executable()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(
        [py_exec, "-m", "streamlit", "run",
         os.path.join("src", "app.py"),
         "--server.headless", "false"],
         cwd=PROJECT_ROOT,
         env=env,
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

    # Free ports from any leftover processes
    free_port(8000)
    free_port(8501)

    proc_names = {}

    if start_both or args.api:
        api_proc = start_api()
        processes.append(api_proc)
        proc_names[api_proc] = "API Server"
        time.sleep(1)  # Let API server grab its port first

        updater_proc = start_updater()
        processes.append(updater_proc)
        proc_names[updater_proc] = "Update Daemon"
        time.sleep(0.5)

    if start_both or args.app:
        app_proc = start_streamlit()
        processes.append(app_proc)
        proc_names[app_proc] = "Streamlit"

    print()
    log("READY", GREEN, "All services are running! 🎉")
    print()
    if start_both or args.api:
        print(f"   🌐 API Server:    {BOLD}http://localhost:8000{RESET}")
        print(f"   📖 API Docs:      {BOLD}http://localhost:8000/docs{RESET}")
        print(f"   📡 Update Daemon: Running background fact-check fetches every 30 mins")
    if start_both or args.app:
        print(f"   🖥️  Dashboard:     {BOLD}http://localhost:8501{RESET}")
    print()
    log("INFO", YELLOW, "Press Ctrl+C to stop all services.\n")

    # Wait for any CRITICAL process to exit (updater is non-critical)
    non_critical = {"Update Daemon"}
    try:
        while True:
            for proc in list(processes):
                retcode = proc.poll()
                if retcode is not None:
                    name = proc_names.get(proc, "Process")
                    if name in non_critical:
                        log("EXIT", YELLOW,
                            f"{name} exited with code {retcode} (non-critical, continuing)")
                        processes.remove(proc)
                    else:
                        log("EXIT", RED,
                            f"{name} exited with code {retcode}")
                        shutdown()
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()

