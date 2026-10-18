#!/usr/bin/env python3
"""Run throughput tests as a category."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = [
    "throughput_tests/test_dbagent_hotpotqa_throughput.py",
    "throughput_tests/test_dbagent_musique_throughput.py",
    "throughput_tests/test_dbagent_throughput_profile.py",
    "throughput_tests/test_throughput_main.py",
    "throughput_tests/test_agentic_reasoning_exec_concurrency.py",
    "throughput_tests/test_musique_agentic_exec_concurrency.py",
    "throughput_tests/test_musique_baseline_concurrency.py",
    "throughput_tests/test_reasoning_baseline_concurrency.py",
    "throughput_tests/test_evadb_concurrency.py",
    "throughput_tests/test_evadb_throughput.py",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for test in TESTS:
            print(test)
        return 0

    failures = []
    for rel in TESTS:
        path = ROOT / rel
        print(f"\n=== Running {rel} ===")
        code = subprocess.run([sys.executable, str(path)], cwd=str(ROOT)).returncode
        if code != 0:
            failures.append(f"{rel} (exit={code})")
    if failures:
        print("\nFAILED:")
        for item in failures:
            print(f"  {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
