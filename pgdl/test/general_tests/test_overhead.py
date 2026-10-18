#!/usr/bin/env python3
"""
Overhead test v2: compare DBAgent vs MorphingDB resource usage per single query.
Categories: Series (CPU), NLP (GPU), Image (GPU).

Isolation:
  - Test script pinned to CPUs 36-39 via os.sched_setaffinity
  - PostgreSQL backend pinned to CPUs 36-39 via taskset
  - PostgreSQL started with CUDA_VISIBLE_DEVICES=2 (only sees GPU 2)
  - No concurrency -- single sequential query per test case

Monitors:
  - CPU: per-query cumulative CPU time (process + children)
  - Memory: process RSS before/after query
  - GPU: per-GPU memory snapshot before/after query
  - GPU utilization: peak during query via polling

Reports both cold-start (first query, models not loaded) and cached (subsequent) overhead.
"""

import os
import sys
import time
import json
import subprocess
import threading
import statistics
from pathlib import Path
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import psutil

# ============================================================
# Configuration
# ============================================================

ISOLATED_CPUS = [36, 37, 38, 39]
ISOLATED_GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "2")

# Runs: 1 cold (model loading) + 3 cached (models already loaded)
COLD_RUNS = 1
CACHED_RUNS = 3

# GPU polling interval during query execution
GPU_POLL_INTERVAL = 0.05

PG_CTL = "/home/why/dbagent/pg_base/bin/pg_ctl"
PG_DATA = "/home/why/dbagent/pg_base/data"
LOG_DIR = Path(__file__).parent.parent / "results"
LOG_DIR.mkdir(exist_ok=True)

# ============================================================
# Test Cases
# ============================================================

TEST_CASES = [
    # --- Series (CPU) ---
    {
        "category": "Series", "name": "slice",
        "dbagent_sql": "SELECT unnest(db_agent_single('series', sub_table.data)) AS score FROM (SELECT * FROM slice_test LIMIT 100) AS sub_table;",
        "morphingdb_sql": "SELECT predict_batch_float8('slice', 'cpu', data) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM slice_test LIMIT 100;",
        "device": "cpu",
    },
    {
        "category": "Series", "name": "swarm",
        "dbagent_sql": "SELECT unnest(db_agent_single('series', sub_table.data)) AS score FROM (SELECT * FROM swarm_test LIMIT 100) AS sub_table;",
        "morphingdb_sql": "SELECT predict_batch_float8('swarm', 'cpu', data) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM swarm_test LIMIT 100;",
        "device": "cpu",
    },
    {
        "category": "Series", "name": "year_predict",
        "dbagent_sql": "SELECT unnest(db_agent_single('series', sub_table.data)) AS score FROM (SELECT * FROM year_predict_test LIMIT 100) AS sub_table;",
        "morphingdb_sql": "SELECT predict_batch_float8('year_predict', 'cpu', data) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM year_predict_test LIMIT 100;",
        "device": "cpu",
    },
    # --- NLP (GPU) ---
    {
        "category": "NLP", "name": "imdb",
        "dbagent_sql": "SELECT unnest(db_agent_single('nlp', sub_table.comment_vec)) AS score FROM (SELECT * FROM imdb_vector_test LIMIT 32) AS sub_table;",
        "morphingdb_sql": "SELECT predict_batch_float8('sst2_vec', 'gpu', comment_vec) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM imdb_vector_test LIMIT 100;",
        "device": "gpu",
    },
    {
        "category": "NLP", "name": "financial",
        "dbagent_sql": "SELECT unnest(db_agent_single('nlp', sub_table.comment_vec)) AS score FROM (SELECT * FROM financial_phrasebank_vector_test LIMIT 32) AS sub_table;",
        "morphingdb_sql": "SELECT predict_batch_float8('finance', 'gpu', comment_vec) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM financial_phrasebank_vector_test LIMIT 100;",
        "device": "gpu",
    },
    {
        "category": "NLP", "name": "sst2",
        "dbagent_sql": "SELECT unnest(db_agent_single('nlp', sub_table.comment_vec)) AS score FROM (SELECT * FROM nlp_vector_test LIMIT 32) AS sub_table;",
        "morphingdb_sql": "SELECT predict_batch_float8('sst2_vec', 'gpu', comment_vec) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM nlp_vector_test LIMIT 100;",
        "device": "gpu",
    },
    # --- Image (GPU) ---
    {
        "category": "Image", "name": "cifar-10",
        "dbagent_sql": "SELECT unnest(db_agent_single('image_classification', sub_table.image_vector)) AS score FROM (SELECT * FROM cifar_image_vector_table LIMIT 32) AS sub_table;",
        "morphingdb_sql": "SELECT predict_batch_float8('googlenet_cifar10', 'gpu', image_vector) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM cifar_image_vector_table LIMIT 100;",
        "device": "gpu",
    },
    {
        "category": "Image", "name": "stanford_dogs",
        "dbagent_sql": "SELECT unnest(db_agent_single('image_classification', sub_table.image_vector)) AS score FROM (SELECT * FROM stanford_dogs_image_vector_table LIMIT 32) AS sub_table;",
        "morphingdb_sql": "SELECT predict_batch_float8('alexnet_stanford_dogs', 'gpu', image_vector) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM stanford_dogs_image_vector_table LIMIT 100;",
        "device": "gpu",
    },
    {
        "category": "Image", "name": "imagenet",
        "dbagent_sql": "SELECT unnest(db_agent_single('image_classification', sub_table.image_vector)) AS score FROM (SELECT * FROM imagenet_image_vector_table LIMIT 32) AS sub_table;",
        "morphingdb_sql": "SELECT predict_batch_float8('defect_vec', 'gpu', image_vector) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM imagenet_image_vector_table LIMIT 100;",
        "device": "gpu",
    },
]


# ============================================================
# GPU snapshot
# ============================================================

def gpu_snapshot():
    """Return {gpu_idx: {'used_mb': float, 'util_pct': float}} for all GPUs."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    )
    snap = {}
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                snap[parts[0]] = {"used_mb": float(parts[1]), "util_pct": float(parts[2])}
    return snap


# ============================================================
# Run query with detailed monitoring
# ============================================================

def run_query(conn, sql, monitor_pid):
    """Execute SQL and return timing + resource deltas."""
    proc = psutil.Process(monitor_pid)

    # Pre-query state
    cpu_before = proc.cpu_times()
    mem_before = proc.memory_info().rss
    gpu_before = gpu_snapshot()

    # Start GPU polling thread
    gpu_peak_util = {}
    gpu_running = True

    def poll_gpu():
        while gpu_running:
            snap = gpu_snapshot()
            for idx, info in snap.items():
                if idx not in gpu_peak_util or info["util_pct"] > gpu_peak_util[idx]:
                    gpu_peak_util[idx] = info["util_pct"]
            time.sleep(GPU_POLL_INTERVAL)

    poll_thread = threading.Thread(target=poll_gpu, daemon=True)
    poll_thread.start()

    # Execute query
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
        error = str(e)[:200]
    elapsed = time.monotonic() - t0

    # Stop GPU polling
    gpu_running = False
    poll_thread.join(timeout=1)

    # Post-query state
    cpu_after = proc.cpu_times()
    mem_after = proc.memory_info().rss
    gpu_after = gpu_snapshot()

    # CPU time delta (user + system) in seconds
    cpu_user_delta = (cpu_after.user - cpu_before.user) if hasattr(cpu_after, 'user') else 0
    cpu_sys_delta = (cpu_after.system - cpu_before.system) if hasattr(cpu_after, 'system') else 0
    cpu_total_delta = cpu_user_delta + cpu_sys_delta

    # Memory delta
    mem_delta_mb = (mem_after - mem_before) / (1024 * 1024)

    # GPU deltas
    gpu_deltas = {}
    for idx in set(list(gpu_before.keys()) + list(gpu_after.keys())):
        before_used = gpu_before.get(idx, {}).get("used_mb", 0)
        after_used = gpu_after.get(idx, {}).get("used_mb", 0)
        before_util = gpu_before.get(idx, {}).get("util_pct", 0)
        after_util = gpu_after.get(idx, {}).get("util_pct", 0)
        peak_util = gpu_peak_util.get(idx, 0)
        gpu_deltas[idx] = {
            "used_before_mb": before_used,
            "used_after_mb": after_used,
            "used_delta_mb": after_used - before_used,
            "util_before_pct": before_util,
            "util_after_pct": after_util,
            "peak_util_pct": peak_util,
        }

    return {
        "elapsed_s": elapsed,
        "cpu_user_s": cpu_user_delta,
        "cpu_sys_s": cpu_sys_delta,
        "cpu_total_s": cpu_total_delta,
        "mem_before_mb": mem_before / (1024 * 1024),
        "mem_after_mb": mem_after / (1024 * 1024),
        "mem_delta_mb": mem_delta_mb,
        "gpu": gpu_deltas,
        "rows": rows,
        "error": error,
    }


def run_test_system(case, system_name, conn, monitor_pid):
    """Run cold + cached queries for one system."""
    sql = case[f"{system_name}_sql"]
    results = {"cold": [], "cached": []}

    # Cold run (first call, models not loaded)
    for i in range(COLD_RUNS):
        r = run_query(conn, sql, monitor_pid)
        results["cold"].append(r)
        time.sleep(2)

    # Cached runs (models already loaded)
    for i in range(CACHED_RUNS):
        r = run_query(conn, sql, monitor_pid)
        results["cached"].append(r)
        if i < CACHED_RUNS - 1:
            time.sleep(2)

    return results


# ============================================================
# Main
# ============================================================

def main():
    print(f"Isolated CPUs: {ISOLATED_CPUS}")
    print(f"CUDA_VISIBLE_DEVICES: {ISOLATED_GPU}")

    # Initial GPU snapshot
    gpu_init = gpu_snapshot()
    print("\nInitial GPU state:")
    for idx, info in sorted(gpu_init.items()):
        print(f"  GPU {idx}: {info['used_mb']:.0f} MB, util={info['util_pct']:.0f}%")
    print()

    import psycopg2
    from morphingdb_test.config import db_config

    conn = psycopg2.connect(**db_config)
    print(f"Connected to postgres")

    cur = conn.cursor()
    cur.execute("SELECT register_process();")
    cur.execute("SELECT pg_backend_pid();")
    monitor_pid = cur.fetchone()[0]
    cur.close()
    print(f"Backend PID: {monitor_pid}")

    # Pin backend to isolated CPUs
    subprocess.run(
        ["taskset", "-p", "-c", ",".join(map(str, ISOLATED_CPUS)), str(monitor_pid)],
        capture_output=True, text=True, timeout=5,
    )
    print(f"Pinned backend to CPUs {ISOLATED_CPUS}")

    results = {}

    for case in TEST_CASES:
        cat = case["category"]
        name = case["name"]
        key = f"{cat}/{name}"
        print(f"\n{'='*60}")
        print(f"Testing: {key} (device: {case['device']})")
        print(f"{'='*60}")

        results[key] = {"category": cat, "name": name, "device": case["device"]}

        # Run morphingdb first to avoid giving dbagent the model cache advantage
        for system in ["morphingdb", "dbagent"]:
            print(f"\n  Running {system}...")
            sys_results = run_test_system(case, system, conn, monitor_pid)
            results[key][system] = sys_results

            # Print cold results
            cold = sys_results["cold"]
            if cold and not cold[0].get("error"):
                c = cold[0]
                print(f"    COLD:  {c['elapsed_s']:.2f}s  CPU={c['cpu_total_s']:.3f}s  MEM={c['mem_after_mb']:.0f}MB (+{c['mem_delta_mb']:+.0f}MB)", end="")
                if case["device"] == "gpu":
                    target_gpu = c["gpu"].get(ISOLATED_GPU, {})
                    print(f"  GPU={target_gpu.get('used_after_mb', 0):.0f}MB (+{target_gpu.get('used_delta_mb', 0):+.0f}MB) peak_util={target_gpu.get('peak_util_pct', 0):.0f}%", end="")
                print()
            elif cold and cold[0].get("error"):
                print(f"    COLD:  ERROR: {cold[0]['error'][:80]}")

            # Print cached results
            cached = sys_results["cached"]
            if cached and not cached[0].get("error"):
                avg_t = statistics.mean([r["elapsed_s"] for r in cached])
                avg_cpu = statistics.mean([r["cpu_total_s"] for r in cached])
                avg_mem = statistics.mean([r["mem_after_mb"] for r in cached])
                avg_mem_delta = statistics.mean([r["mem_delta_mb"] for r in cached])
                print(f"    CACHED: avg {avg_t:.3f}s  CPU={avg_cpu:.3f}s  MEM={avg_mem:.0f}MB (delta={avg_mem_delta:+.0f}MB)", end="")
                if case["device"] == "gpu":
                    target_gpu_deltas = [r["gpu"].get(ISOLATED_GPU, {}).get("used_delta_mb", 0) for r in cached]
                    target_gpu_peaks = [r["gpu"].get(ISOLATED_GPU, {}).get("peak_util_pct", 0) for r in cached]
                    avg_gpu_delta = statistics.mean(target_gpu_deltas) if target_gpu_deltas else 0
                    avg_gpu_peak = statistics.mean(target_gpu_peaks) if target_gpu_peaks else 0
                    print(f"  GPU delta={avg_gpu_delta:+.0f}MB peak_util={avg_gpu_peak:.0f}%", end="")
                print()

    # Final GPU snapshot
    gpu_final = gpu_snapshot()
    conn.close()

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = LOG_DIR / f"overhead_v2_{ts}.json"
    output_data = {
        "metadata": {
            "isolated_cpus": ISOLATED_CPUS,
            "cuda_visible_devices": ISOLATED_GPU,
            "gpu_initial": gpu_init,
            "gpu_final": gpu_final,
            "timestamp": ts,
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary tables
    print_summary(results, gpu_init, gpu_final)


def print_summary(results, gpu_init, gpu_final):
    print(f"\n{'='*120}")
    print("COLD-START OVERHEAD (first query, models loaded from scratch)")
    print(f"{'='*120}")

    for cat in ["Series", "NLP", "Image"]:
        print(f"\n--- {cat} ---")
        print(f"{'Dataset':<16} {'System':<14} {'Time(s)':>8} {'CPU(s)':>8} {'MEM(MB)':>10} {'MEM_delta':>10} {'GPUMem(MB)':>12} {'GPU_delta':>10} {'GPU_peak_util%':>14}")
        print("-" * 110)

        for key, data in sorted(results.items()):
            if data["category"] != cat:
                continue
            name = data["name"]
            for system in ["dbagent", "morphingdb"]:
                sys_data = data.get(system, {})
                cold_runs = sys_data.get("cold", [])
                if cold_runs and not cold_runs[0].get("error"):
                    c = cold_runs[0]
                    t = c["elapsed_s"]
                    cpu = c["cpu_total_s"]
                    mem = c["mem_after_mb"]
                    mem_d = c["mem_delta_mb"]
                    if data["device"] == "gpu":
                        g = c["gpu"].get(ISOLATED_GPU, {})
                        gpu_mem = g.get("used_after_mb", 0)
                        gpu_d = g.get("used_delta_mb", 0)
                        gpu_util = g.get("peak_util_pct", 0)
                        print(f"{name:<16} {system:<14} {t:>8.2f} {cpu:>8.3f} {mem:>10.0f} {mem_d:>10.0f} {gpu_mem:>12.0f} {gpu_d:>10.0f} {gpu_util:>14.0f}")
                    else:
                        print(f"{name:<16} {system:<14} {t:>8.2f} {cpu:>8.3f} {mem:>10.0f} {mem_d:>10.0f} {'N/A':>12} {'N/A':>10} {'N/A':>14}")

    print(f"\n{'='*120}")
    print("CACHED OVERHEAD (models already loaded, avg of 3 runs)")
    print(f"{'='*120}")

    for cat in ["Series", "NLP", "Image"]:
        print(f"\n--- {cat} ---")
        print(f"{'Dataset':<16} {'System':<14} {'Time(s)':>8} {'CPU(s)':>8} {'MEM(MB)':>10} {'MEM_delta':>10} {'GPUMem_delta':>14} {'GPU_peak_util%':>14}")
        print("-" * 90)

        for key, data in sorted(results.items()):
            if data["category"] != cat:
                continue
            name = data["name"]
            for system in ["dbagent", "morphingdb"]:
                sys_data = data.get(system, {})
                cached_runs = sys_data.get("cached", [])
                if cached_runs and not cached_runs[0].get("error"):
                    avg_t = statistics.mean([r["elapsed_s"] for r in cached_runs])
                    avg_cpu = statistics.mean([r["cpu_total_s"] for r in cached_runs])
                    avg_mem = statistics.mean([r["mem_after_mb"] for r in cached_runs])
                    avg_mem_d = statistics.mean([r["mem_delta_mb"] for r in cached_runs])
                    if data["device"] == "gpu":
                        gpu_deltas = [r["gpu"].get(ISOLATED_GPU, {}).get("used_delta_mb", 0) for r in cached_runs]
                        gpu_peaks = [r["gpu"].get(ISOLATED_GPU, {}).get("peak_util_pct", 0) for r in cached_runs]
                        avg_gpu_d = statistics.mean(gpu_deltas) if gpu_deltas else 0
                        avg_gpu_p = statistics.mean(gpu_peaks) if gpu_peaks else 0
                        print(f"{name:<16} {system:<14} {avg_t:>8.3f} {avg_cpu:>8.3f} {avg_mem:>10.0f} {avg_mem_d:>10.0f} {avg_gpu_d:>14.0f} {avg_gpu_p:>14.0f}")
                    else:
                        print(f"{name:<16} {system:<14} {avg_t:>8.3f} {avg_cpu:>8.3f} {avg_mem:>10.0f} {avg_mem_d:>10.0f} {'N/A':>14} {'N/A':>14}")

    print(f"\nGPU state change during entire test:")
    for idx in sorted(set(list(gpu_init.keys()) + list(gpu_final.keys()))):
        before = gpu_init.get(idx, {}).get("used_mb", 0)
        after = gpu_final.get(idx, {}).get("used_mb", 0)
        print(f"  GPU {idx}: {before:.0f} -> {after:.0f} MB (delta: {after - before:+.0f} MB)")

    print(f"\n{'='*120}")


if __name__ == "__main__":
    try:
        os.sched_setaffinity(0, ISOLATED_CPUS)
        print(f"[INFO] Pinned test script (PID {os.getpid()}) to CPUs {ISOLATED_CPUS}")
    except Exception as e:
        print(f"[WARN] Could not pin CPU affinity: {e}")

    main()
