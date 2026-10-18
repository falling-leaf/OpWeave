#!/usr/bin/env python3
"""
Measure GPU memory usage of DBAgent during reasoning tasks.
Tests n = 1, 5, 10, 20 rows, cold-start + 3 cached runs each.
Pinned to CPUs 21-39 to avoid interfering with timing tasks on CPUs 1-20.
"""

import os
import sys
import time
import json
import subprocess
import threading
import statistics
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import psutil
import psycopg2
from morphingdb_test.config import db_config

from tools.test_result_paths import GENERAL_RESULTS_DIR

ISOLATED_CPUS = list(range(21, 40))
GPU_POLL_INTERVAL = 0.1
COLD_RUNS = 1
CACHED_RUNS = 3
N_VALUES = [1, 5, 10, 20]

LOG_DIR = Path(GENERAL_RESULTS_DIR)
LOG_DIR.mkdir(exist_ok=True)


def gpu_snapshot():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    )
    snap = {}
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                snap[parts[0]] = {
                    "used_mb": float(parts[1]),
                    "total_mb": float(parts[2]),
                    "util_pct": float(parts[3]),
                }
    return snap


def run_query(conn, sql, monitor_pid):
    proc = psutil.Process(monitor_pid)

    gpu_before = gpu_snapshot()
    gpu_peak = {idx: info["used_mb"] for idx, info in gpu_before.items()}
    stop_poll = False

    def poll():
        while not stop_poll:
            snap = gpu_snapshot()
            for idx, info in snap.items():
                if info["used_mb"] > gpu_peak.get(idx, 0):
                    gpu_peak[idx] = info["used_mb"]
            time.sleep(GPU_POLL_INTERVAL)

    t = threading.Thread(target=poll, daemon=True)
    t.start()

    t0 = time.monotonic()
    error = None
    rows = 0
    try:
        cur = conn.cursor()
        cur.execute("SELECT register_process();")
        cur.execute(sql)
        rows = len(cur.fetchall())
        cur.close()
    except Exception as e:
        error = str(e)[:300]
    elapsed = time.monotonic() - t0

    stop_poll = True
    t.join(timeout=1)

    gpu_after = gpu_snapshot()

    gpu_stats = {}
    for idx in set(list(gpu_before.keys()) + list(gpu_after.keys())):
        b = gpu_before.get(idx, {})
        a = gpu_after.get(idx, {})
        gpu_stats[idx] = {
            "before_mb": b.get("used_mb", 0),
            "after_mb":  a.get("used_mb", 0),
            "peak_mb":   gpu_peak.get(idx, 0),
            "delta_mb":  a.get("used_mb", 0) - b.get("used_mb", 0),
            "total_mb":  a.get("total_mb", 0),
        }

    return {
        "elapsed_s": elapsed,
        "rows": rows,
        "error": error,
        "gpu": gpu_stats,
    }


def main():
    try:
        os.sched_setaffinity(0, set(ISOLATED_CPUS))
        print(f"[INFO] Pinned to CPUs {ISOLATED_CPUS[0]}-{ISOLATED_CPUS[-1]}")
    except Exception as e:
        print(f"[WARN] CPU affinity: {e}")

    print("\nInitial GPU state:")
    gpu_init = gpu_snapshot()
    for idx, info in sorted(gpu_init.items()):
        print(f"  GPU {idx}: {info['used_mb']:.0f}/{info['total_mb']:.0f} MB  util={info['util_pct']:.0f}%")

    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute("SELECT register_process();")
    cur.execute("SELECT pg_backend_pid();")
    backend_pid = cur.fetchone()[0]
    cur.close()
    print(f"\nBackend PID: {backend_pid}")

    subprocess.run(
        ["taskset", "-cp", f"{ISOLATED_CPUS[0]}-{ISOLATED_CPUS[-1]}", str(backend_pid)],
        capture_output=True,
    )

    all_results = {}

    for n in N_VALUES:
        sql = (
            f"SELECT unnest(db_agent_single('reasoning', sub_table.reasoning_vec)) AS score "
            f"FROM (SELECT * FROM reasoning_vector_test ORDER BY id LIMIT {n}) AS sub_table;"
        )
        print(f"\n{'='*60}")
        print(f"n = {n}  |  SQL: ...LIMIT {n}...")
        print(f"{'='*60}")

        runs = {"cold": [], "cached": []}

        for phase in ["cold"] * COLD_RUNS + ["cached"] * CACHED_RUNS:
            r = run_query(conn, sql, backend_pid)
            runs[phase].append(r)
            status = f"ERROR: {r['error']}" if r["error"] else f"ok ({r['rows']} rows, {r['elapsed_s']:.2f}s)"
            phase_label = "COLD  " if phase == "cold" else "CACHED"
            gpu_info = "  ".join(
                f"GPU{idx}: before={v['before_mb']:.0f} peak={v['peak_mb']:.0f} after={v['after_mb']:.0f} MB"
                for idx, v in sorted(r["gpu"].items())
            )
            print(f"  [{phase_label}] {status}")
            print(f"           {gpu_info}")
            time.sleep(3)

        all_results[n] = runs

    conn.close()
    gpu_final = gpu_snapshot()

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SUMMARY — DBAgent reasoning GPU memory (MB)")
    print(f"{'='*80}")
    print(f"{'n':<5} {'Phase':<8} {'GPU idx':<8} {'Before':>8} {'Peak':>8} {'After':>8} {'Delta':>8} {'Time(s)':>9}")
    print("-" * 70)

    for n in N_VALUES:
        for phase_key, label in [("cold", "COLD"), ("cached", "CACHED(avg)")]:
            phase_runs = all_results[n][phase_key]
            if not phase_runs or phase_runs[0]["error"]:
                print(f"{n:<5} {label:<8}  ERROR")
                continue

            avg_elapsed = statistics.mean(r["elapsed_s"] for r in phase_runs)
            # aggregate GPU stats across runs
            all_gpu_idx = sorted(phase_runs[0]["gpu"].keys())
            for idx in all_gpu_idx:
                before = statistics.mean(r["gpu"][idx]["before_mb"] for r in phase_runs)
                peak   = statistics.mean(r["gpu"][idx]["peak_mb"]   for r in phase_runs)
                after  = statistics.mean(r["gpu"][idx]["after_mb"]  for r in phase_runs)
                delta  = statistics.mean(r["gpu"][idx]["delta_mb"]  for r in phase_runs)
                print(f"{n:<5} {label:<12} GPU{idx:<4} {before:>8.0f} {peak:>8.0f} {after:>8.0f} {delta:>+8.0f} {avg_elapsed:>9.2f}")

    print(f"\nGPU state change (entire test):")
    for idx in sorted(set(list(gpu_init.keys()) + list(gpu_final.keys()))):
        b = gpu_init.get(idx, {}).get("used_mb", 0)
        a = gpu_final.get(idx, {}).get("used_mb", 0)
        print(f"  GPU {idx}: {b:.0f} -> {a:.0f} MB  (delta {a-b:+.0f} MB)")

    # ── Save JSON ─────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = LOG_DIR / f"reasoning_gpu_{ts}.json"
    save = {
        "metadata": {"gpu_init": gpu_init, "gpu_final": gpu_final, "timestamp": ts},
        "results": {str(n): v for n, v in all_results.items()},
    }
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
