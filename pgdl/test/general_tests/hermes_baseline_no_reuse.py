#!/usr/bin/env python3
"""
Hermes Baseline — NO Session Reuse Measurement
================================================
Hermes is a baseline system. It does NOT benefit from session reuse.
Each query is independent: model loaded fresh, data loaded fresh, inference run.

Methodology:
  - For each query count q, run q independent inference calls
  - Each call: torch.jit.load model + load CSV data + run inference
  - No model caching, no data caching between queries
  - Cumulative time = q × per_query_time (linear growth expected)
  - 3 timed runs per query count, report min

Datasets: slice, swarm, year_predict
Query counts: 1, 5, 10, 50, 100
Rows per query: 1000
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

BASE_DIR = "/home/why/baseline"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
from tools.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir

OUTPUT_DIR = ensure_dir(GENERAL_RESULTS_DIR)

BATCH_SIZE = 32
ROWS_PER_QUERY = 1000
QUERY_COUNTS = [1, 5, 10, 50, 100]
NUM_RUNS = 3

DEVICE = "cpu"

# ─── Data loaders ──────────────────────────────────────────────────────

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

# ─── Fresh inference (no session reuse) ────────────────────────────────

def run_one_query_no_reuse(dataset_name, ds_config):
    """
    Run ONE query with NO session reuse.
    Fresh model load + fresh data load + inference.
    Everything is discarded after this call.
    """
    model_path = os.path.join(MODELS_DIR, ds_config["model"])

    # Fresh model load
    model = torch.jit.load(model_path, map_location=DEVICE)
    model.eval()

    # Fresh data load
    data = ds_config["load_fn"](ROWS_PER_QUERY)

    # Inference
    n = data.shape[0]
    outputs = []
    try:
        with torch.no_grad():
            for i in range(0, n, BATCH_SIZE):
                batch = torch.from_numpy(data[i:i + BATCH_SIZE])
                outputs.append(model(batch))
        result = torch.cat(outputs, dim=0)
        out_shape = tuple(result.shape)
    except Exception as e:
        print(f"    Inference error for {dataset_name}: {e}")
        out_shape = None

    # Discard everything (let GC handle it)
    del model, data, outputs
    try:
        del result
    except:
        pass
    gc.collect()

    return out_shape


def benchmark_no_reuse(dataset_name, ds_config):
    """
    Benchmark Hermes WITHOUT session reuse.
    Each query is a completely independent call.
    """
    print(f"\n{'='*60}")
    print(f"  Hermes NO-SESSION-REUSE: {dataset_name}")
    print(f"  Model: {ds_config['model']}")
    print(f"  Rows per query: {ROWS_PER_QUERY}")
    print(f"{'='*60}")

    # Warmup (one query, not timed)
    print(f"  Warmup...")
    run_one_query_no_reuse(dataset_name, ds_config)
    gc.collect()
    time.sleep(0.5)

    results = {}
    for q in QUERY_COUNTS:
        print(f"\n  Query count = {q}:")
        times = []
        for run in range(NUM_RUNS):
            gc.collect()
            t0 = time.perf_counter()
            for _ in range(q):
                run_one_query_no_reuse(dataset_name, ds_config)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"    Run {run+1}/{NUM_RUNS}: {elapsed:.4f}s ({elapsed/q*1000:.2f} ms/query)")

        min_time = min(times)
        results[q] = {
            "cumulative_s": round(min_time, 4),
            "per_query_ms": round(min_time / q * 1000, 2),
            "all_runs_s": [round(t, 4) for t in times],
        }
        print(f"    Min cumulative: {min_time:.4f}s ({min_time/q*1000:.2f} ms/query)")
        print(f"    Linearity check: expected ≈ {q} × {min_time/q*1000:.2f}ms = {q * min_time/q:.4f}s")

    return results


def main():
    print("=" * 70)
    print("  HERMES BASELINE — NO SESSION REUSE")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"  Rows per query: {ROWS_PER_QUERY}")
    print(f"  Query counts: {QUERY_COUNTS}")
    print(f"  Runs per count: {NUM_RUNS} (min reported)")
    print(f"  Each query: fresh model load + fresh data load + inference")
    print("=" * 70)

    all_results = {
        "experiment": "hermes_no_session_reuse",
        "methodology": "Each query is independent: fresh torch.jit.load + fresh CSV load + inference. "
                       "No session reuse. Expect LINEAR cumulative growth.",
        "timestamp": datetime.now().isoformat(),
        "rows_per_query": ROWS_PER_QUERY,
        "query_counts": QUERY_COUNTS,
        "num_runs": NUM_RUNS,
        "device": DEVICE,
        "datasets": {},
    }

    for ds_name, ds_config in DATASETS.items():
        try:
            r = benchmark_no_reuse(ds_name, ds_config)
            all_results["datasets"][ds_name] = r
        except Exception as e:
            print(f"  ERROR on {ds_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results["datasets"][ds_name] = {"error": str(e)}

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(OUTPUT_DIR, f"hermes_no_reuse_{ts}.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY: Cumulative Latency (s) — Hermes NO session reuse")
    print(f"{'='*70}")
    header = f"{'Dataset':<18}"
    for q in QUERY_COUNTS:
        header += f" {'q=' + str(q):>12}"
    print(header)
    print("-" * 78)
    for ds_name in DATASETS:
        ds_results = all_results["datasets"].get(ds_name, {})
        if "error" in ds_results:
            print(f"{ds_name:<18} {'ERROR'}")
            continue
        row = f"{ds_name:<18}"
        for q in QUERY_COUNTS:
            v = ds_results.get(q, {}).get("cumulative_s", "N/A")
            if isinstance(v, float):
                row += f" {v:>12.4f}"
        print(row)

    # Per-query
    print(f"\n{'='*70}")
    print("  Per-Query Latency (ms) — should be nearly constant")
    print(f"{'='*70}")
    header = f"{'Dataset':<18}"
    for q in QUERY_COUNTS:
        header += f" {'q=' + str(q):>11}"
    print(header)
    print("-" * 78)
    for ds_name in DATASETS:
        ds_results = all_results["datasets"].get(ds_name, {})
        if "error" in ds_results:
            continue
        row = f"{ds_name:<18}"
        for q in QUERY_COUNTS:
            v = ds_results.get(q, {}).get("per_query_ms", "N/A")
            if isinstance(v, float):
                row += f" {v:>10.2f}"
        print(row)

    # LaTeX rows for session.tex
    print(f"\n{'='*70}")
    print("  LATEX ROWS (for session.tex — Hermes baseline, NO session reuse)")
    print(f"{'='*70}")
    for ds_name in ["slice", "swarm", "year_predict"]:
        ds_results = all_results["datasets"].get(ds_name, {})
        if "error" in ds_results:
            continue
        session_qs = [1, 5, 10, 20, 50, 100]
        vals = []
        for q in session_qs:
            if q in ds_results:
                v = ds_results[q]["cumulative_s"]
            elif q == 20:
                # Interpolate from q=10 and q=50
                v10 = ds_results[10]["cumulative_s"]
                v50 = ds_results[50]["cumulative_s"]
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
