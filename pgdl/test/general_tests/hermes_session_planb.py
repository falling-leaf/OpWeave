#!/usr/bin/env python3
"""
Hermes Session Reuse — Plan B Measurement (fixed)
===================================================
Two hermes rows for session.tex:

  hermes (exec)  — model loaded once per Python session, reused across q queries.
                   Sub-linear growth; represents in-process Python caching.

  hermes (full)  — each query is a fresh Hermes agent call (LLM codegen + exec).
                   Strictly linear: cumulative(q) = q × agent_time.
                   Codegen overhead recorded explicitly.

CPU pinning: 21-40 only.
"""

import os
import sys
import json
import time
import gc
import subprocess
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_DIR   = "/home/why/baseline"
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
from tools.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir

OUTPUT_DIR = ensure_dir(GENERAL_RESULTS_DIR)
HERMES_CLI = "/home/why/.local/bin/hermes"

BATCH_SIZE     = 32
ROWS_PER_QUERY = 1000
SESSION_QUERIES    = [1, 5, 10, 20, 50, 100]
EXEC_QUERY_COUNTS  = [1, 5, 10, 50, 100]   # 20 will be interpolated
NUM_EXEC_RUNS  = 3
NUM_AGENT_RUNS = 3
AGENT_TIMEOUT  = 120  # seconds per agent call
CPU_AFFINITY   = "21-40"

DATASETS = {
    "slice": {
        "model": "slice.pt",
        "load_fn": lambda n: pd.read_csv(
            os.path.join(DATA_DIR, "series", "slice", "slice_localization_data.csv"),
            nrows=n).iloc[:, 1:-1].values.astype(np.float32),
    },
    "swarm": {
        "model": "swarm.pt",
        "load_fn": lambda n: pd.read_csv(
            os.path.join(DATA_DIR, "series", "swarm", "Swarm_Behaviour.csv"),
            nrows=n).drop(columns=["Swarm_Behaviour"]).values.astype(np.float32),
    },
    "year_predict": {
        "model": "year_predict.pt",
        "load_fn": lambda n: pd.read_csv(
            os.path.join(DATA_DIR, "series", "yead_predict", "YearPredictionMSD.csv"),
            nrows=n).drop(columns=["value90"]).values.astype(np.float32),
    },
}

HERMES_PROMPTS = {
    "slice": (
        "Load the TorchScript model at /home/why/baseline/models/slice.pt on CPU. "
        "Read the first 1000 rows of "
        "/home/why/baseline/data/series/slice/slice_localization_data.csv, "
        "use columns index 1 to -1 as float32 features. "
        "Run batched inference with batch_size=32 and print the output shape."
    ),
    "swarm": (
        "Load the TorchScript model at /home/why/baseline/models/swarm.pt on CPU. "
        "Read the first 1000 rows of "
        "/home/why/baseline/data/series/swarm/Swarm_Behaviour.csv, "
        "drop the column 'Swarm_Behaviour', use remaining columns as float32 features. "
        "Run batched inference with batch_size=32 and print the output shape."
    ),
    "year_predict": (
        "Load the TorchScript model at /home/why/baseline/models/year_predict.pt on CPU. "
        "Read the first 1000 rows of "
        "/home/why/baseline/data/series/yead_predict/YearPredictionMSD.csv, "
        "drop the column 'value90', use remaining 90 columns as float32 features. "
        "Run batched inference with batch_size=32 and print the output shape."
    ),
}

# ─── Utilities ───────────────────────────────────────────────────────────────

def pin_cpus(cpus):
    try:
        subprocess.run(["taskset", "-cp", cpus, str(os.getpid())],
                       check=True, capture_output=True)
        print(f"[pin] PID {os.getpid()} → CPUs {cpus}", flush=True)
    except Exception as e:
        print(f"[pin] WARNING: {e}", flush=True)


def infer(model, data):
    outputs = []
    with torch.no_grad():
        for i in range(0, len(data), BATCH_SIZE):
            batch = torch.from_numpy(data[i:i + BATCH_SIZE])
            outputs.append(model(batch))
    return torch.cat(outputs, dim=0)


def run_hermes(ds_name):
    """Single Hermes agent call without --tui. Returns elapsed seconds."""
    prompt = HERMES_PROMPTS[ds_name]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            [HERMES_CLI, "-z", prompt, "--yolo"],
            capture_output=True, text=True,
            timeout=AGENT_TIMEOUT,
            cwd=BASE_DIR,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        elapsed = time.perf_counter() - t0
        return elapsed, r.returncode == 0
    except subprocess.TimeoutExpired:
        return time.perf_counter() - t0, False
    except Exception as e:
        return time.perf_counter() - t0, False

# ─── Part 1: exec measurement ─────────────────────────────────────────────────

def measure_exec(ds_name, ds_cfg):
    print(f"\n--- exec: {ds_name} ---", flush=True)
    model = torch.jit.load(os.path.join(MODELS_DIR, ds_cfg["model"]), map_location="cpu")
    model.eval()
    data = ds_cfg["load_fn"](ROWS_PER_QUERY)
    print(f"  data shape: {data.shape}", flush=True)

    # warmup
    for _ in range(3):
        infer(model, data)
    gc.collect()

    raw = {}
    for q in EXEC_QUERY_COUNTS:
        runs = []
        for _ in range(NUM_EXEC_RUNS):
            gc.collect()
            t0 = time.perf_counter()
            for _q in range(q):
                infer(model, data)
            runs.append(time.perf_counter() - t0)
        raw[q] = min(runs)
        print(f"  q={q:>3}: {raw[q]:.4f}s ({raw[q]/q*1000:.2f} ms/q, min of {NUM_EXEC_RUNS})", flush=True)

    # interpolate q=20
    raw[20] = raw[10] + (raw[50] - raw[10]) * (20 - 10) / (50 - 10)
    print(f"  q= 20: {raw[20]:.4f}s (interpolated)", flush=True)

    return {
        "cumulative_s": {q: round(raw[q], 4) for q in SESSION_QUERIES},
        "per_query_ms": {q: round(raw[q] / q * 1000, 2) for q in SESSION_QUERIES},
    }

# ─── Part 2: agent (full pipeline) measurement ───────────────────────────────

def measure_agent(ds_name):
    print(f"\n--- agent: {ds_name} ---", flush=True)
    times = []
    for i in range(NUM_AGENT_RUNS):
        elapsed, ok = run_hermes(ds_name)
        status = "ok" if ok else "FAIL"
        print(f"  run {i+1}/{NUM_AGENT_RUNS}: {elapsed:.3f}s [{status}]", flush=True)
        times.append((elapsed, ok))

    # prefer successful runs; fall back to all
    good = [t for t, ok in times if ok]
    vals = good if good else [t for t, _ in times]
    min_t = min(vals)
    avg_t = sum(vals) / len(vals)

    cumulative = {q: round(q * min_t, 4) for q in SESSION_QUERIES}
    print(f"  min_agent: {min_t:.3f}s, avg: {avg_t:.3f}s", flush=True)

    return {
        "min_agent_time_s":  round(min_t, 4),
        "avg_agent_time_s":  round(avg_t, 4),
        "all_runs_s":        [round(t, 4) for t, _ in times],
        "cumulative_s":      cumulative,
        "per_query_ms":      {q: round(min_t * 1000, 2) for q in SESSION_QUERIES},
    }

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    pin_cpus(CPU_AFFINITY)

    print("=" * 70, flush=True)
    print("  HERMES SESSION REUSE — PLAN B", flush=True)
    print(f"  {datetime.now().isoformat()}", flush=True)
    print(f"  rows/query={ROWS_PER_QUERY}  exec_runs={NUM_EXEC_RUNS}  agent_runs={NUM_AGENT_RUNS}", flush=True)
    print("=" * 70, flush=True)

    all_results = {
        "experiment": "hermes_session_planb",
        "timestamp": datetime.now().isoformat(),
        "rows_per_query": ROWS_PER_QUERY,
        "session_queries": SESSION_QUERIES,
        "cpu_affinity": CPU_AFFINITY,
        "datasets": {},
    }

    # ── exec pass (all datasets) ────────────────────────────────────────
    print("\n\n=== PART 1: exec (in-session model reuse) ===", flush=True)
    exec_results = {}
    for ds_name, ds_cfg in DATASETS.items():
        exec_results[ds_name] = measure_exec(ds_name, ds_cfg)

    # ── agent pass (all datasets) ───────────────────────────────────────
    print("\n\n=== PART 2: agent (full pipeline per query) ===", flush=True)
    agent_results = {}
    for ds_name in DATASETS:
        agent_results[ds_name] = measure_agent(ds_name)

    # ── Assemble final results ──────────────────────────────────────────
    for ds_name in DATASETS:
        exec_q1 = exec_results[ds_name]["cumulative_s"][1]
        agent   = agent_results[ds_name]
        codegen = round(agent["min_agent_time_s"] - exec_q1, 4)
        agent["codegen_overhead_s"] = codegen
        agent["exec_per_query_s"]   = exec_q1

        all_results["datasets"][ds_name] = {
            "hermes_exec": exec_results[ds_name],
            "hermes_full": agent,
        }

    # ── Save ────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_json = os.path.join(OUTPUT_DIR, f"hermes_session_planb_{ts}.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON saved → {out_json}", flush=True)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print("  CODEGEN OVERHEAD", flush=True)
    print(f"{'='*70}", flush=True)
    for ds_name in DATASETS:
        fd = all_results["datasets"][ds_name]["hermes_full"]
        pct = fd["codegen_overhead_s"] / fd["min_agent_time_s"] * 100
        print(f"  {ds_name:<16}: total={fd['min_agent_time_s']:.3f}s | "
              f"codegen={fd['codegen_overhead_s']:.3f}s ({pct:.1f}%) | "
              f"exec={fd['exec_per_query_s']:.4f}s", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("  CUMULATIVE LATENCY (s)", flush=True)
    print(f"{'='*70}", flush=True)
    hdr = f"{'Dataset / Row':<30}"
    for q in SESSION_QUERIES:
        hdr += f" {'q='+str(q):>9}"
    print(hdr, flush=True)
    print("-" * (30 + 10 * len(SESSION_QUERIES)), flush=True)
    for ds_name in DATASETS:
        ds = all_results["datasets"][ds_name]
        for label, key in [("hermes (exec)", "hermes_exec"), ("hermes (full)", "hermes_full")]:
            tag = f"{ds_name} / {label}"
            row = f"{tag:<30}"
            for q in SESSION_QUERIES:
                v = ds[key]["cumulative_s"][q]
                row += f" {v:>9.4f}"
            print(row, flush=True)
        print(flush=True)

    # ── LaTeX rows ───────────────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print("  LATEX ROWS FOR session.tex", flush=True)
    print(f"{'='*70}", flush=True)
    for ds_name in DATASETS:
        ds = all_results["datasets"][ds_name]
        for label, key in [("hermes (exec)", "hermes_exec"), ("hermes (full)", "hermes_full")]:
            vals = " & ".join(f"{ds[key]['cumulative_s'][q]:.4f}" for q in SESSION_QUERIES)
            print(f"    & \\textit{{{label}}} & {vals} \\\\", flush=True)
        print(flush=True)

    print("Done!", flush=True)
    return out_json


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
