#!/usr/bin/env python3
"""
Run the supported PGDL smoke/regression tests after rebuilding/restarting
the extension.

Excluded on purpose:
- throughput tests under `throughput_tests/`
- CPU/GPU load or under-load tests under `scheduling_tests/`
- long-running benchmark scripts in `general_tests/`
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

SUPPORTED_TESTS = [
    "general_tests/test_breakdown_summary.py",
    "general_tests/test_correctness.py",
    "general_tests/test_hotpotqa_breakdown.py",
    "general_tests/test_inference_overhead_ratio.py",
    "general_tests/test_overhead.py",
    "general_tests/test_overhead_comparison.py",
    "general_tests/test_reasoning_samples.py",
    "general_tests/test_single.py",
    "general_tests/test_sql_timing.py",
]

LONG_RUNNING_REFERENCE_TESTS = [
    "general_tests/test_agent_timing.py",
    "general_tests/test_hermes_overhead.py",
    "general_tests/test_reasoning_independent.py",
    "general_tests/test_reasoning_python_bench.py",
    "general_tests/test_reasoning_sequential.py",
    "general_tests/test_skew.py",
    "general_tests/test_timing_table.py",
    "scheduling_tests/test_dbaagent_scheduling_decision.py",
    "scheduling_tests/test_fault_tolerance_cifar10.py",
    "scheduling_tests/load_test/run_table3.py",
]

DEFAULT_TIMEOUT_SECONDS = 180


def run_script(script: str, timeout: int) -> int:
    path = ROOT / script
    print(f"\n=== Running {script} ===")
    try:
        proc = subprocess.run([sys.executable, str(path)], cwd=str(ROOT), timeout=timeout)
        return proc.returncode
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {script} exceeded {timeout}s")
        return 124


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List supported smoke tests without running them.")
    parser.add_argument(
        "--list-long",
        action="store_true",
        help="List longer non-throughput scripts tracked separately from the smoke set.",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    if args.list:
        for script in SUPPORTED_TESTS:
            print(script)
        return 0

    if args.list_long:
        for script in LONG_RUNNING_REFERENCE_TESTS:
            print(script)
        return 0

    failures: list[str] = []
    for script in SUPPORTED_TESTS:
        code = run_script(script, args.timeout)
        if code != 0:
            failures.append(f"{script} (exit={code})")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("\nAll supported non-throughput, non-load smoke/regression tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
