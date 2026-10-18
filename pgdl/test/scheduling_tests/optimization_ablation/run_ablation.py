#!/usr/bin/env python3
"""Measure DBAgent latency with optimization on or off under Table 3 loads.

One invocation is one cell family: one optimization state, one workload,
three load phases (low / medium / high). The binary under test must already
be the one installed for ``--optimization``. This script does not compile,
does not install, and does not restart PostgreSQL.

Benchmark protocol matches the existing scheduling tests:
  pin to CPUs 0-19, 1 discarded warmup, N timed runs, report the minimum.

Usage (after the matching binary is installed):

    python run_ablation.py --optimization on  --workload year_predict
    python run_ablation.py --optimization off --workload cifar10 --repetitions 3
    python run_ablation.py --optimization on  --workload hotpotqa --phases low,high

Results:
    pgdl/test/results/scheduling/optimization_ablation/
        <workload>_opt_<on|off>_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST_ROOT = HERE.parents[1]  # .../pgdl/test
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TEST_ROOT))

from load_harness import (  # noqa: E402
    ALLOWED_CPU_MASK,
    CpuLoadManager,
    CpuMonitor,
    GpuLoadManager,
    get_gpu_util,
    pin_backend,
    pin_current_process,
)
from support.test_result_paths import SCHEDULING_RESULTS_DIR, ensure_dir  # noqa: E402
from workloads import WORKLOADS, get_workload  # noqa: E402

DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "user": "why",
    "dbname": "postgres",
    "password": "123456",
}

RESULTS_DIR = ensure_dir(Path(SCHEDULING_RESULTS_DIR) / "optimization_ablation")

COOLDOWN_SECONDS = 60
CPU_STABILIZATION_SECONDS = 3
GPU_STABILIZATION_SECONDS = 8
WARMUP_RUNS = 3
TIMED_RUNS = 3


def _run_query(cur, query: str) -> dict:
    start = time.perf_counter()
    try:
        cur.execute(query)
        rows = cur.fetchall()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "latency_ms": round(elapsed_ms, 1),
            "status": "success",
            "rows_returned": len(rows),
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        err = str(exc)
        status = "timeout" if "timeout" in err.lower() or "canceling" in err.lower() else "error"
        return {
            "latency_ms": round(elapsed_ms, 1),
            "status": status,
            "rows_returned": 0,
            "error": err,
        }


def _start_load(workload: dict, phase: dict):
    if workload["kind"] == "cpu":
        manager = CpuLoadManager()
        manager.start(
            phase["n_workers"],
            phase.get("burn_rounds", 64),
            phase.get("pin_cpu", 0),
            phase.get("extra_workers", 0),
            phase.get("extra_pin_cpu", 20),
       
            phase.get("wake_gap", 0),
            phase.get("duty", 100),
        )
        gpu_workers = phase.get("gpu_workers", 0)
        if gpu_workers:
            # swarm's GPU inference is too short for the CPU-20 wake worker to
            # move it, so its high phase also parks matmul workers on GPU 0.
            # They never touch the CPU path, so the off row is unaffected.
            manager.gpu = GpuLoadManager()
            manager.gpu.start(gpu_workers, phase.get("gpu_mem_gb", 2.0), oom_style=True)
        return manager
    manager = GpuLoadManager()
    manager.start(phase["n_workers"], phase["mem_gb"], oom_style=phase.get("oom_style", False))
    return manager


def _sample_load(workload: dict, monitor: CpuMonitor | None):
    if workload["kind"] == "cpu":
        return {"cpu_pct": monitor.get_usage()}
    util, mem_used = get_gpu_util()
    return {"gpu_util_pct": util, "gpu_mem_mb": mem_used}


def run_phase(conn_factory, workload: dict, phase: dict, cooldown: int, timed_runs: int) -> dict:
    name = phase["name"]
    print(f"\n{'=' * 70}")
    print(f"  PHASE: {name.upper()}  workload={workload['key']}")
    print(f"{'=' * 70}")

    manager = _start_load(workload, phase)
    monitor = CpuMonitor() if workload["kind"] == "cpu" else None
    settle = CPU_STABILIZATION_SECONDS if workload["kind"] == "cpu" else GPU_STABILIZATION_SECONDS
    print(f"  Stabilizing for {settle}s...")
    time.sleep(settle)
    if monitor is not None:
        monitor.warmup()
        time.sleep(0.2)

    before = _sample_load(workload, monitor)
    print(f"  Load before queries: {before}")

    import psycopg2

    conn = conn_factory(psycopg2)
    cur = conn.cursor()
    query = workload["query"].format(row_count=workload["row_count"])

    print(f"\n  Warmup runs (discarded)...")
    warmup = None
    for w in range(WARMUP_RUNS):
        warmup = _run_query(cur, query)
        print(f"    Warmup {w + 1}: {warmup['latency_ms']:.1f}ms [{warmup['status']}]")

    print(f"\n  Timed runs ({timed_runs})...")
    timed_results = []
    samples = []
    for i in range(timed_runs):
        sample_before = _sample_load(workload, monitor)
        result = _run_query(cur, query)
        sample_after = _sample_load(workload, monitor)
        samples.append({"before": sample_before, "after": sample_after})
        mark = "OK" if result["status"] == "success" else "!!"
        print(
            f"  {i + 1:4d} | {result['latency_ms']:12.1f} ms | "
            f"rows={result['rows_returned']:6d} | {mark}"
        )
        timed_results.append(result)

    conn.close()
    manager.stop()

    if cooldown > 0:
        print(f"\n  Cooling down for {cooldown}s...")
        time.sleep(cooldown)

    successful = [r["latency_ms"] for r in timed_results if r["status"] == "success"]
    min_lat = min(successful) if successful else None
    avg_lat = sum(successful) / len(successful) if successful else None
    print(f"  Min latency: {min_lat:.1f} ms" if min_lat is not None else "  Min latency: N/A")

    return {
        "phase": name,
        "phase_config": {k: v for k, v in phase.items() if k != "name"},
        "load_before_queries": before,
        "load_samples": samples,
        "warmup": warmup,
        "timed_results": timed_results,
        "min_latency_ms": round(min_lat, 1) if min_lat is not None else None,
        "avg_latency_ms": round(avg_lat, 1) if avg_lat is not None else None,
        "min_latency_s": round(min_lat / 1000.0, 3) if min_lat is not None else None,
    }


def connect(psycopg2, statement_timeout: str):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT pg_backend_pid()")
    backend_pid = cur.fetchone()[0]
    pin_backend(backend_pid)
    affinity = os.sched_getaffinity(backend_pid)
    # CPU inference runs on CPU 0. A GPU call is moved to CPU 20 inside
    # infer_batch_internal, because the CUDA driver sync queues for seconds
    # when it shares CPU 0 with the busy-wait workers.
    if not set(affinity) <= set(range(21)):
        raise SystemExit(
            f"backend {backend_pid} affinity is {sorted(affinity)}, expected CPUs 0-20"
        )
    print(f"  backend {backend_pid} pinned to CPUs 0-19")
    cur.execute("SELECT register_process()")
    cur.execute(f"SET statement_timeout = '{statement_timeout}'")
    cur.close()
    return conn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization", choices=("on", "off"), required=True)
    parser.add_argument(
        "--workload",
        choices=tuple(w["key"] for w in WORKLOADS),
        required=True,
    )
    parser.add_argument("--phases", default="low,medium,high",
                        help="Comma-separated subset of low,medium,high")
    parser.add_argument("--repetitions", type=int, default=TIMED_RUNS)
    parser.add_argument("--cooldown", type=int, default=COOLDOWN_SECONDS)
    parser.add_argument("--row-count", type=int, default=None,
                        help="Override the Table 3 row count. Default keeps the paper n.")
    parser.add_argument("--save", action="store_true", default=True)
    args = parser.parse_args()

    workload = dict(get_workload(args.workload))
    if args.row_count is not None:
        workload["row_count"] = args.row_count

    wanted = [name.strip() for name in args.phases.split(",") if name.strip()]
    known = {phase["name"]: phase for phase in workload["phases"]}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise SystemExit(f"unknown phases {unknown}; choices: {sorted(known)}")
    phases = [known[name] for name in wanted]

    pin_current_process()

    print("DBAgent optimization ablation (Table 3 extra row)")
    print(f"  optimization:    {args.optimization}")
    print(f"  workload:        {workload['label']}  n={workload['row_count']}")
    print(f"  default device:  {workload['default_device_when_off']} when optimization is off")
    print(f"  phases:          {', '.join(wanted)}")
    print(f"  timed runs:      {args.repetitions} (report minimum)")
    print(f"  allowed CPUs:    {ALLOWED_CPU_MASK}")
    print("  This process does not install a binary. Confirm the running")
    print(f"  pgdl.so was built with ADD_OPTIMIZATION_LOGIC={args.optimization}.")

    overall_start = time.time()
    phase_results = []
    for phase in phases:
        phase_results.append(
            run_phase(
                lambda pg, timeout=workload["statement_timeout"]: connect(pg, timeout),
                workload,
                phase,
                args.cooldown,
                args.repetitions,
            )
        )
    total_time = time.time() - overall_start

    print(f"\n{'=' * 70}")
    print(f"  {workload['key']}  optimization={args.optimization}")
    print(f"{'=' * 70}")
    print(f"  {'Phase':<8s} | {'Min (s)':>10s} | {'Min (ms)':>12s}")
    for result in phase_results:
        seconds = f"{result['min_latency_s']:.3f}" if result["min_latency_s"] is not None else "N/A"
        millis = f"{result['min_latency_ms']:.1f}" if result["min_latency_ms"] is not None else "N/A"
        print(f"  {result['phase']:<8s} | {seconds:>10s} | {millis:>12s}")
    print(f"  Total time: {total_time / 60:.1f} min")

    payload = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "optimization_ablation_table3",
        "optimization": args.optimization,
        "workload": workload["key"],
        "paper_column": workload["paper_column"],
        "default_device_when_off": workload["default_device_when_off"],
        "config": {
            "row_count": workload["row_count"],
            "task": workload["task"],
            "kind": workload["kind"],
            "query": workload["query"].format(row_count=workload["row_count"]),
            "warmup_runs": WARMUP_RUNS,
            "timed_runs": args.repetitions,
            "cooldown_seconds": args.cooldown,
            "allowed_cpus": ALLOWED_CPU_MASK,
            "reported_stat": "minimum of successful timed runs",
        },
        "total_time_seconds": round(total_time, 1),
        "phases": phase_results,
    }

    if args.save:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            RESULTS_DIR,
            f"{workload['key']}_opt_{args.optimization}_{stamp}.json",
        )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\n  Results saved to: {path}")


if __name__ == "__main__":
    main()
