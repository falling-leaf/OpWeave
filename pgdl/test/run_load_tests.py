#!/usr/bin/env python3
"""Run CPU/GPU load and under-load tests as a category."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = [
    "scheduling_tests/load_test/run_table3.py",
    "scheduling_tests/test_dbaagent_scheduling_decision.py",
    "scheduling_tests/test_dbagent_reasoning_under_load_all_gpu.py",
    "scheduling_tests/test_fault_tolerance_cifar10.py",
    "scheduling_tests/test_hermes_agent_cifar10_all_loads.py",
    "scheduling_tests/test_hermes_agent_cifar10_split.py",
    "scheduling_tests/test_hermes_agent_year_predict_under_load.py",
    "scheduling_tests/test_hermes_cifar10_all_loads_single_run.py",
    "scheduling_tests/test_hermes_cifar10_under_load.py",
    "scheduling_tests/test_hermes_reasoning_under_load.py",
    "scheduling_tests/test_hermes_reasoning_under_load_all_gpu.py",
    "scheduling_tests/test_hermes_slice_under_load.py",
    "scheduling_tests/test_hermes_year_predict_under_load.py",
    "scheduling_tests/test_reasoning_cpu_load_dbagent.py",
    "scheduling_tests/test_reasoning_cpu_load_hermes.py",
    "scheduling_tests/test_reasoning_cpu_load_morphingdb.py",
    "scheduling_tests/test_slice_under_load.py",
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
