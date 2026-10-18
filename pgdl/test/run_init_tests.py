#!/usr/bin/env python3
"""Run init/setup tests as a category."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "init_tests" / "run_tests.py"

if __name__ == "__main__":
    raise SystemExit(subprocess.run([sys.executable, str(SCRIPT)], cwd=str(ROOT)).returncode)
