#!/usr/bin/env python3
"""
Hermes Agent-Based Session Reuse — Method 2 Backup Measurement
================================================================
Uses the Hermes CLI (hermes -q "prompt") to measure full agent pipeline
latency for each query. This represents the "no session reuse" baseline.

Usage:
    python3 hermes_agent_reuse_backup.py [--dry-run]

In dry-run mode, prints the commands that would be run without executing them.
In live mode, runs hermes CLI commands and measures E2E time.

NOTE: This script requires the Hermes CLI to be installed and configured.
Each query calls the Hermes agent, which involves LLM API calls and may
be slow/expensive. Use with caution.
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime

HERMES_CLI = "/home/why/.local/bin/hermes"
from tools.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir

OUTPUT_DIR = ensure_dir(GENERAL_RESULTS_DIR)

# Prompt templates for each dataset
PROMPTS = {
    "slice": (
        "Run inference using the slice.pt model on 1000 rows from "
        "slice_localization_data.csv. Use the benchmark.py inference pattern: "
        "torch.jit.load the model, load data with pandas, run batched inference "
        "with batch_size=32 on CPU. Measure and report the latency."
    ),
    "swarm": (
        "Run inference using the swarm.pt model on 1000 rows from "
        "Swarm_Behaviour.csv. Use the benchmark.py inference pattern: "
        "torch.jit.load the model, load data (drop Swarm_Behaviour column), "
        "run batched inference with batch_size=32 on CPU. Report the latency."
    ),
    "year_predict": (
        "Run inference using the year_predict.pt model on 1000 rows from "
        "YearPredictionMSD.csv. Use the benchmark.py inference pattern: "
        "torch.jit.load the model, load data (drop value90 column), "
        "run batched inference with batch_size=32 on CPU. Report the latency."
    ),
}

QUERY_COUNTS = [1, 5, 10]  # Reduced for agent-based measurement (slow)
TIMEOUT_PER_QUERY = 120  # seconds


def run_hermes_query(prompt, timeout=TIMEOUT_PER_QUERY):
    """Run a single Hermes agent query and return wall-clock time."""
    cmd = [HERMES_CLI, "-q", prompt]
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd="/home/why/baseline",
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        )
        elapsed = time.perf_counter() - t0
        success = result.returncode == 0
        return {
            "elapsed_s": round(elapsed, 4),
            "success": success,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-500:] if result.stdout else "",
            "stderr_tail": result.stderr[-500:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return {
            "elapsed_s": round(elapsed, 4),
            "success": False,
            "returncode": -1,
            "stdout_tail": "",
            "stderr_tail": "TIMEOUT",
        }


def main():
    dry_run = "--dry-run" in sys.argv

    print("=" * 70)
    print("  HERMES AGENT-BASED SESSION REUSE — Method 2 Backup")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    if dry_run:
        print("\nCommands that would be run:")
        for ds_name, prompt in PROMPTS.items():
            for q in QUERY_COUNTS:
                print(f"\n  [{ds_name}] q={q}:")
                print(f"    hermes -q \"{prompt[:80]}...\"")
                if q > 1:
                    print(f"    (repeated {q} times)")
        print("\nDone (dry run).")
        return

    all_results = {
        "experiment": "hermes_agent_reuse_backup",
        "method": "method2_full_agent_per_query",
        "timestamp": datetime.now().isoformat(),
        "query_counts": QUERY_COUNTS,
        "datasets": {},
    }

    for ds_name, prompt in PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        ds_results = {}
        for q in QUERY_COUNTS:
            print(f"\n  Query count = {q}:")
            times = []
            for i in range(q):
                print(f"    Query {i+1}/{q}...", end=" ", flush=True)
                r = run_hermes_query(prompt)
                times.append(r)
                status = "OK" if r["success"] else f"FAIL(rc={r['returncode']})"
                print(f"{r['elapsed_s']:.4f}s {status}")

            cumulative = sum(t["elapsed_s"] for t in times)
            all_ok = all(t["success"] for t in times)
            ds_results[q] = {
                "cumulative_s": round(cumulative, 4),
                "per_query_ms": round(cumulative / q * 1000, 2),
                "all_ok": all_ok,
                "individual_runs": times,
            }
            print(f"    Cumulative: {cumulative:.4f}s ({cumulative/q*1000:.2f} ms/query)")

        all_results["datasets"][ds_name] = ds_results

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(
        OUTPUT_DIR,
        f"hermes_agent_reuse_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
