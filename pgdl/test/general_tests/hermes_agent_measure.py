#!/usr/bin/env python3
"""
Hermes Agent Session Reuse — Actual Agent Measurement
=======================================================
Measures Hermes agent E2E time by actually calling `hermes -z "prompt"`
and timing the full pipeline (LLM + code gen + execution + result processing).

Method 1 (Primary, for session.tex):
  - q=1 = min of 3 agent runs (full pipeline)
  - q>1 = q1_agent_time + (q-1) × pure_inference_time_per_query
  - Pure inference time from hermes_session_reuse.py measurement

Method 2 (Backup):
  - q queries = q × full_agent_time_per_query (no session reuse)
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime

HERMES_CLI = "/home/why/.local/bin/hermes"
BASELINE_DIR = "/home/why/baseline"
from tools.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir

OUTPUT_DIR = ensure_dir(GENERAL_RESULTS_DIR)

NUM_RUNS = 3  # min of 3 runs per dataset
TIMEOUT_PER_RUN = 180  # seconds

QUERY_COUNTS = [1, 5, 10, 50, 100]

# Prompt templates — includes dataset-specific model/data paths
PROMPTS = {
    "slice": (
        "Load the model at /home/why/baseline/models/slice.pt using torch.jit.load on CPU. "
        "Load 1000 rows from /home/why/baseline/data/series/slice/slice_localization_data.csv, "
        "take columns 1 to -1 as float32 features. Run batched inference with batch_size=32 "
        "and report output shape and precise inference time."
    ),
    "swarm": (
        "Load the model at /home/why/baseline/models/swarm.pt using torch.jit.load on CPU. "
        "Load 1000 rows from /home/why/baseline/data/series/swarm/Swarm_Behaviour.csv, "
        "drop the 'Swarm_Behaviour' column, use remaining 2400 columns as float32 features. "
        "Run batched inference with batch_size=32 and report output shape and precise inference time."
    ),
    "year_predict": (
        "Load the model at /home/why/baseline/models/year_predict.pt using torch.jit.load on CPU. "
        "Load 1000 rows from /home/why/baseline/data/series/yead_predict/YearPredictionMSD.csv, "
        "drop the 'value90' column, use remaining 90 columns as float32 features. "
        "Run batched inference with batch_size=32 and report output shape and precise inference time."
    ),
}

# Pure inference per-query times (from hermes_session_reuse.py measurement)
# These are the stable per-query costs after agent overhead is paid
PURE_INFERENCE_PER_QUERY_S = {
    "slice": 0.005,        # ~5ms per query on CPU
    "swarm": 0.007,        # ~7ms per query on CPU
    "year_predict": 0.001, # ~1ms per query on CPU
}


def run_hermes_agent(dataset, prompt):
    """Run a single Hermes agent query and return wall-clock time."""
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [HERMES_CLI, "-z", prompt, "--tui"],
            capture_output=True, text=True,
            timeout=TIMEOUT_PER_RUN,
            cwd=BASELINE_DIR,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        )
        elapsed = time.perf_counter() - t0
        success = result.returncode == 0
        return {
            "elapsed_s": round(elapsed, 4),
            "success": success,
            "returncode": result.returncode,
            "stdout_last_500": result.stdout[-500:] if result.stdout else "",
            "stderr_last_500": result.stderr[-500:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return {
            "elapsed_s": round(elapsed, 4),
            "success": False,
            "returncode": -1,
            "stdout_last_500": "",
            "stderr_last_500": "TIMEOUT",
        }


def main():
    print("=" * 70)
    print("  HERMES AGENT SESSION REUSE — ACTUAL AGENT MEASUREMENT")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"  Runs per dataset: {NUM_RUNS} (min reported)")
    print(f"  Query counts: {QUERY_COUNTS}")
    print("=" * 70)

    all_results = {
        "experiment": "hermes_agent_session_reuse_measured",
        "methodology": "Actual Hermes agent calls via `hermes -z`. "
                       "q=1 = min of 3 agent runs. "
                       "q>1 = agent_time + (q-1) × pure_inference_time.",
        "timestamp": datetime.now().isoformat(),
        "num_runs": NUM_RUNS,
        "query_counts": QUERY_COUNTS,
        "datasets": {},
    }

    for ds_name in ["slice", "swarm", "year_predict"]:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        prompt = PROMPTS[ds_name]
        times = []

        for run in range(NUM_RUNS):
            print(f"\n  Run {run+1}/{NUM_RUNS}...", end=" ", flush=True)
            r = run_hermes_agent(ds_name, prompt)
            times.append(r)
            status = "✓" if r["success"] else f"✗ (rc={r['returncode']})"
            print(f"{r['elapsed_s']:.3f}s {status}")

        successful = [t for t in times if t["success"]]
        if not successful:
            print(f"  ERROR: All {NUM_RUNS} runs failed for {ds_name}!")
            all_results["datasets"][ds_name] = {"error": "all runs failed", "runs": times}
            continue

        min_time = min(t["elapsed_s"] for t in successful)
        avg_time = sum(t["elapsed_s"] for t in successful) / len(successful)
        print(f"\n  Min: {min_time:.3f}s | Avg: {avg_time:.3f}s | All: {[t['elapsed_s'] for t in successful]}")

        pure_per_query = PURE_INFERENCE_PER_QUERY_S[ds_name]

        # Method 1: Code reuse (code generated once, reused across queries)
        method1 = {
            "description": "Code generated once (first query), reused for subsequent queries",
            "agent_time_q1_s": round(min_time, 4),
            "pure_inference_per_query_s": pure_per_query,
            "all_agent_runs_s": [t["elapsed_s"] for t in successful],
            "avg_agent_time_s": round(avg_time, 4),
            "cumulative": {},
        }
        for q in QUERY_COUNTS:
            cum = min_time + (q - 1) * pure_per_query
            method1["cumulative"][q] = {
                "cumulative_s": round(cum, 4),
                "per_query_ms": round(cum / q * 1000, 2),
            }

        # Method 2: Full agent per query (no session reuse)
        method2 = {
            "description": "Each query is a full agent call (no session reuse)",
            "agent_time_per_query_s": round(min_time, 4),
            "cumulative": {},
        }
        for q in QUERY_COUNTS:
            method2["cumulative"][q] = {
                "cumulative_s": round(q * min_time, 4),
                "per_query_ms": round(min_time * 1000, 2),
            }

        all_results["datasets"][ds_name] = {
            "method1_code_reuse": method1,
            "method2_full_agent": method2,
        }

        # Print summary
        print(f"\n  Method 1 (Code Reuse — Primary):")
        for q in QUERY_COUNTS:
            c = method1["cumulative"][q]
            print(f"    q={q:>3}: cumulative={c['cumulative_s']:.4f}s ({c['per_query_ms']:.2f} ms/query)")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(OUTPUT_DIR, f"hermes_agent_measured_{ts}.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Generate LaTeX rows
    print(f"\n{'='*70}")
    print("  LATEX ROWS (for session.tex)")
    print(f"{'='*70}")
    for ds_name in ["slice", "swarm", "year_predict"]:
        m1 = all_results["datasets"][ds_name]["method1_code_reuse"]
        # session.tex uses: 1, 5, 10, 20, 50, 100
        session_qs = [1, 5, 10, 20, 50, 100]
        vals = []
        for q in session_qs:
            if q in m1["cumulative"]:
                v = m1["cumulative"][q]["cumulative_s"]
            elif q == 20:
                # Interpolate
                v10 = m1["cumulative"][10]["cumulative_s"]
                v50 = m1["cumulative"][50]["cumulative_s"]
                v = v10 + (v50 - v10) * (20 - 10) / (50 - 10)
                v = round(v, 4)
            else:
                v = 0.0
            vals.append(v)
        row = f"    & hermes & {vals[0]:.4f} & {vals[1]:.4f} & {vals[2]:.4f} & {vals[3]:.4f} & {vals[4]:.4f} & {vals[5]:.4f} \\\\"
        print(row)

    print("\nDone!")


if __name__ == "__main__":
    main()
