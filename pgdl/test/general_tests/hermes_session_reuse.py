#!/usr/bin/env python3
"""
Hermes Session Reuse Benchmark on Series Datasets
====================================================
Measures cumulative latency of repeated inference calls within a single
Python session (model loaded once, reused across queries).

Datasets: slice, swarm, year_predict
Query counts: 1, 5, 10, 50, 100
Rows per query: 1000

Methodology:
  - Model loaded once at session start (outside timing)
  - Data (1000 rows) loaded once per dataset (outside timing)
  - For each query count q, run inference q times sequentially on the
    same data batch, measuring cumulative wall-clock time
  - 3 timed runs per query count, report min
  - Warmup runs before each query count to ensure stable GPU/CPU state
"""

import os
import sys
import time
import json
import gc
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# ─── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = "/home/why/baseline"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
from tools.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir

OUTPUT_DIR = ensure_dir(GENERAL_RESULTS_DIR)

BATCH_SIZE = 32
ROWS_PER_QUERY = 1000
QUERY_COUNTS = [1, 5, 10, 50, 100]
NUM_RUNS = 3  # report min of 3 runs

DEVICE_CPU = "cpu"

# ─── Dataset definitions ──────────────────────────────────────────────

def load_slice_data(n_rows):
    path = os.path.join(DATA_DIR, "series", "slice", "slice_localization_data.csv")
    df = pd.read_csv(path, nrows=n_rows)
    return df.iloc[:, 1:-1].values.astype(np.float32)

def load_swarm_data(n_rows):
    path = os.path.join(DATA_DIR, "series", "swarm", "Swarm_Behaviour.csv")
    df = pd.read_csv(path, nrows=n_rows)
    return df.drop(columns=["Swarm_Behaviour"]).values.astype(np.float32)

def load_year_predict_data(n_rows):
    path = os.path.join(DATA_DIR, "series", "yead_predict", "YearPredictionMSD.csv")
    df = pd.read_csv(path, nrows=n_rows)
    return df.drop(columns=["value90"]).values.astype(np.float32)

DATASETS = {
    "slice": {
        "model": "slice.pt",
        "load_fn": load_slice_data,
    },
    "swarm": {
        "model": "swarm.pt",
        "load_fn": load_swarm_data,
    },
    "year_predict": {
        "model": "year_predict.pt",
        "load_fn": load_year_predict_data,
    },
}

# ─── Batched inference (CPU, matches Hermes benchmark.py) ─────────────

def infer_series(model, features_np, batch_size=BATCH_SIZE):
    """Tabular inference: numpy array -> output tensor. CPU."""
    n = features_np.shape[0]
    outputs = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(features_np[i:i + batch_size])
            outputs.append(model(batch))
    return torch.cat(outputs, dim=0)


def benchmark_session_reuse(dataset_name, ds_config):
    """Run session reuse benchmark for one dataset."""
    model_path = os.path.join(MODELS_DIR, ds_config["model"])
    load_fn = ds_config["load_fn"]

    print(f"\n{'='*60}")
    print(f"Session Reuse Benchmark: {dataset_name}")
    print(f"  Model: {ds_config['model']}")
    print(f"  Rows per query: {ROWS_PER_QUERY}")
    print(f"  Query counts: {QUERY_COUNTS}")
    print(f"{'='*60}")

    # Step 1: Load model (session init, outside timing)
    print(f"[1/3] Loading model: {ds_config['model']} on {DEVICE_CPU}...")
    model = torch.jit.load(model_path, map_location=DEVICE_CPU)
    model.eval()
    print(f"  Model loaded.")

    # Step 2: Load data (outside timing, same data for all queries)
    print(f"[2/3] Loading {ROWS_PER_QUERY} rows of data...")
    data = load_fn(ROWS_PER_QUERY)
    print(f"  Data shape: {data.shape}")

    # Step 3: Warmup
    print(f"[3/3] Warming up...")
    for _ in range(3):
        _ = infer_series(model, data)
    gc.collect()
    print(f"  Warmup complete.")

    # Step 4: Benchmark at each query count
    results = {}
    for q in QUERY_COUNTS:
        print(f"\n  Query count = {q}:")
        times = []
        for run in range(NUM_RUNS):
            gc.collect()
            t0 = time.perf_counter()
            for _ in range(q):
                _ = infer_series(model, data)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"    Run {run+1}/{NUM_RUNS}: {elapsed:.4f}s")

        min_time = min(times)
        results[q] = {
            "cumulative_s": round(min_time, 4),
            "per_query_ms": round(min_time / q * 1000, 2),
            "all_runs_s": [round(t, 4) for t in times],
        }
        print(f"    Min cumulative: {min_time:.4f}s ({min_time/q*1000:.2f} ms/query)")

    return results


def main():
    print("=" * 70)
    print("  HERMES SESSION REUSE BENCHMARK")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"  Rows per query: {ROWS_PER_QUERY}")
    print(f"  Query counts: {QUERY_COUNTS}")
    print(f"  Runs per count: {NUM_RUNS} (min reported)")
    print(f"  Device: {DEVICE_CPU}")
    print(f"  Batch size: {BATCH_SIZE}")
    print("=" * 70)

    all_results = {
        "experiment": "hermes_session_reuse",
        "timestamp": datetime.now().isoformat(),
        "rows_per_query": ROWS_PER_QUERY,
        "query_counts": QUERY_COUNTS,
        "num_runs": NUM_RUNS,
        "device": DEVICE_CPU,
        "batch_size": BATCH_SIZE,
        "datasets": {},
    }

    for ds_name, ds_config in DATASETS.items():
        try:
            r = benchmark_session_reuse(ds_name, ds_config)
            all_results["datasets"][ds_name] = r
        except Exception as e:
            print(f"  ERROR on {ds_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results["datasets"][ds_name] = {"error": str(e)}

    # ─── Save results ─────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(
        OUTPUT_DIR,
        f"hermes_session_reuse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # ─── Summary table ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY: Cumulative Latency (s)")
    print(f"{'='*70}")
    header = f"{'Dataset':<18}"
    for q in QUERY_COUNTS:
        header += f" {'q=' + str(q):>12}"
    print(header)
    print("-" * (18 + 13 * len(QUERY_COUNTS)))
    for ds_name in DATASETS:
        ds_results = all_results["datasets"].get(ds_name, {})
        if "error" in ds_results:
            print(f"{ds_name:<18} {'ERROR: ' + ds_results['error'][:40]}")
            continue
        row = f"{ds_name:<18}"
        for q in QUERY_COUNTS:
            v = ds_results.get(q, {}).get("cumulative_s", "N/A")
            if isinstance(v, float):
                row += f" {v:>12.4f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # Per-query latency
    print(f"\n{'='*70}")
    print("  SUMMARY: Per-Query Latency (ms)")
    print(f"{'='*70}")
    header = f"{'Dataset':<18}"
    for q in QUERY_COUNTS:
        header += f" {'q=' + str(q):>12}"
    print(header)
    print("-" * (18 + 13 * len(QUERY_COUNTS)))
    for ds_name in DATASETS:
        ds_results = all_results["datasets"].get(ds_name, {})
        if "error" in ds_results:
            print(f"{ds_name:<18} {'ERROR'}")
            continue
        row = f"{ds_name:<18}"
        for q in QUERY_COUNTS:
            v = ds_results.get(q, {}).get("per_query_ms", "N/A")
            if isinstance(v, float):
                row += f" {v:>11.2f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    print("\nDone!")


if __name__ == "__main__":
    main()
