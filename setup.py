#!/usr/bin/env python3
"""
Pet Breed Recognition - One-click setup script
Supports Windows / Linux / macOS
"""

import sys, os, subprocess, platform, shutil

def print_banner():
    print("=" * 50)
    print("  Pet Breed Recognition & Care Advisor")
    print("  One-Click Setup & Launch")
    print("=" * 50)

def check_python():
    ver = sys.version_info
    print(f"[1/4] Checking Python version... current: {ver.major}.{ver.minor}.{ver.micro}")
    if ver.major < 3 or (ver.major == 3 and ver.minor < 10):
        print(f"\nERROR: Python 3.10+ required, got {ver.major}.{ver.minor}")
        sys.exit(1)
    print("  OK\n")

def setup_venv():
    base = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(base, ".venv")
    print("[2/4] Setting up virtual environment...")
    if os.path.exists(venv_dir):
        print("  Virtual env already exists, skipping")
    else:
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        print("  Virtual env created")
    is_win = platform.system() == "Windows"
    pip = os.path.join(venv_dir, "Scripts" if is_win else "bin", "pip" + (".exe" if is_win else ""))
    py = os.path.join(venv_dir, "Scripts" if is_win else "bin", "python" + (".exe" if is_win else ""))
    return pip, py

def install_deps(pip):
    print("[3/4] Installing dependencies...")
    req = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    mirrors = [
        "https://pypi.tuna.tsinghua.edu.cn/simple",
        "https://pypi.org/simple",
    ]
    for m in mirrors:
        print(f"  Trying: {m}")
        r = subprocess.run([pip, "install", "-r", req, "-i", m])
        if r.returncode == 0:
            print("  Done\n")
            return
        print("  Failed, trying next...")
    print("  All mirrors failed. Check your network.")
    sys.exit(1)

def show_menu(py):
    print("=" * 50)
    print("[4/4] Setup complete!")
    print("=" * 50)
    print()
    print("  [1] Launch app now")
    print("  [2] Install Ollama model + Launch")
    print("  [3] Exit")
    print()
    choice = input("Choose (1/2/3): ").strip()
    app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

    if choice == "1":
        print("\nLaunching... Open http://localhost:8501 (Ctrl+C to stop)\n")
        subprocess.run([py, "-m", "streamlit", "run", app])
    elif choice == "2":
        if shutil.which("ollama") or shutil.which("ollama.exe"):
            print("\nPulling qwen2.5:1.5b model...")
            subprocess.run(["ollama", "pull", "qwen2.5:1.5b"])
        else:
            print("\nOllama not found. Install from: https://ollama.com/")
            print("Then run: ollama pull qwen2.5:1.5b")
        print("\nLaunching app...")
        subprocess.run([py, "-m", "streamlit", "run", app])
    else:
        print("\nTo launch manually: streamlit run app.py")

if __name__ == "__main__":
    print_banner()
    check_python()
    pip, py = setup_venv()
    install_deps(pip)
    show_menu(py)