#!/usr/bin/env python3
"""MorphingDB and Hermes under the same load as the ablation.

year_predict pins its contention to CPU 0 (low=1, medium=4, high=12
/tmp/spin_load workers). cifar10 uses the same GPU 0 matmul workers as
run_ablation.py (low=2x2GB, medium=3x2GB, high=8x1.5GB oom-style). The older
under_load scripts are a different experiment.

MorphingDB keeps the orchestration-default device: year_predict on CPU 0,
cifar10 on gpu:0. Hermes excludes the model load and adds the SQL fetch,
which the DBAgent query time includes.

Usage:

    python compare_baselines.py
    python compare_baselines.py --workload cifar10
    python compare_baselines.py --systems morphingdb --phases high
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.test_result_paths import SCHEDULING_RESULTS_DIR, ensure_dir  # noqa: E402

from load_harness import CpuLoadManager, CpuMonitor, GpuLoadManager  # noqa: E402
from workloads import get_workload  # noqa: E402

HERE = Path(__file__).resolve().parent
HERMES_SCRIPTS = {
    "year_predict": HERE.parent / "baseline_code" / "hermes_year_predict.py",
    "cifar10": HERE.parent / "baseline_code" / "hermes_cifar10.py",
}
RESULTS_DIR = Path(ensure_dir(Path(SCHEDULING_RESULTS_DIR) / "optimization_ablation"))

DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "user": "why",
    "dbname": "postgres",
    "password": "123456",
}

MORPHINGDB_QUERIES = {
    "year_predict": (
        "SELECT predict_batch_float8('year_predict', 'cpu', data) "
        "OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) "
        "FROM year_predict_test LIMIT {row_count};"
    ),
    "cifar10": (
        "SELECT predict_batch_float8('googlenet_cifar10', 'gpu:0', image_vector) "
        "OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) "
        "FROM cifar_image_vector_table LIMIT {row_count};"
    ),
}
# year_predict's CPU inference shares CPU 0 with the spin workers.
# cifar10 stays on the default mask: the GPU call is driven from inside
# the backend, and pinning it to CPU 0 would only add a CPU stall.
BACKEND_CPU = {"year_predict": "0", "cifar10": "0-19"}
STATEMENT_TIMEOUT = {"year_predict": "300s", "cifar10": "1800s"}
HERMES_TIMEOUT = {"year_predict": 600, "cifar10": 1800}

COOLDOWN_SECONDS = 60
STABILIZATION_SECONDS = {"year_predict": 3, "cifar10": 8}
WARMUP_RUNS = 1
TIMED_RUNS = 3


def _run_morphingdb(workload: str, row_count: int) -> dict:
    import psycopg2

    start = time.perf_counter()
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT pg_backend_pid()")
        backend_pid = cur.fetchone()[0]
        # predict_batch_float8 does not pass through infer_batch_internal, so
        # the backend has to be pinned here.
        subprocess.run(
            ["taskset", "-p", "-c", BACKEND_CPU[workload], str(backend_pid)],
            capture_output=True, timeout=5,
        )
        cur.execute("SELECT register_process()")
        cur.execute(f"SET statement_timeout = '{STATEMENT_TIMEOUT[workload]}'")
        cur.execute(MORPHINGDB_QUERIES[workload].format(row_count=row_count))
        rows = cur.fetchall()
        conn.close()
        return {
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "status": "success",
            "rows_returned": len(rows),
        }
    except Exception as exc:
        message = str(exc)
        status = "timeout" if "timeout" in message.lower() or "canceling" in message.lower() else "error"
        return {
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "status": status,
            "rows_returned": 0,
            "error": message[:300],
        }


def _run_hermes(workload: str, row_count: int) -> dict:
    env = os.environ.copy()
    if workload == "year_predict":
        env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = ["taskset", "-c", "0"]
    else:
        cmd = ["taskset", "-c", "0-19"]
    cmd += [
        sys.executable, str(HERMES_SCRIPTS[workload]),
        "--row-count", str(row_count), "--repetitions", "1", "--json-output",
    ]
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=HERMES_TIMEOUT[workload], env=env,
        )
    except subprocess.TimeoutExpired:
        return {"latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "status": "timeout", "error": "timeout 600s"}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "status": "error",
            "error": (proc.stderr or "no output").strip()[:300],
        }
    result = json.loads(proc.stdout.strip())
    # avg is the inference loop only. load_s is the SQL fetch plus tensor
    # build, which the DBAgent query time includes, so the comparable number
    # is the sum.
    return {
        "latency_ms": round((result["avg"] + result["load_s"]) * 1000, 1),
        "inference_ms": round(result["avg"] * 1000, 1),
        "data_load_ms": round(result["load_s"] * 1000, 1),
        "status": "success",
    }


RUNNERS = {"morphingdb": _run_morphingdb, "hermes": _run_hermes}


def _start_load(workload: str, phase: dict):
    if workload == "year_predict":
        # Same phases as run_ablation.py: low=1 spin, medium=2 spins on CPU 0,
        # high=4 spins on CPU 0 plus the wake worker on CPU 20. MorphingDB and
        # Hermes stay on CPU, so the wake worker does not change their path,
        # but the load has to match the DBAgent row.
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
        return manager
    manager = GpuLoadManager()
    manager.start(phase["n_workers"], phase["mem_gb"], oom_style=phase.get("oom_style", False))
    return manager


def run_phase(system: str, workload: str, phase: dict, row_count: int, cooldown: int, timed_runs: int) -> dict:
    name = phase["name"]
    n_workers = phase["n_workers"]
    print(f"\n{'=' * 70}")
    print(f"  {system.upper()}  {workload}  PHASE {name.upper()}  ({n_workers} workers)")
    print(f"{'=' * 70}")

    manager = _start_load(workload, phase)
    time.sleep(STABILIZATION_SECONDS[workload])

    if workload == "year_predict":
        monitor = CpuMonitor()
        monitor.warmup()
        time.sleep(0.2)
        print(f"  CPU load before queries: {monitor.get_usage():.1f}%")

    runner = RUNNERS[system]
    print("  Warmup (discarded)...")
    warmup = runner(workload, row_count)
    print(f"    {warmup['latency_ms']:.1f} ms [{warmup['status']}]")

    timed = []
    for i in range(timed_runs):
        result = runner(workload, row_count)
        mark = "OK" if result["status"] == "success" else "!!"
        print(f"    {i + 1} | {result['latency_ms']:10.1f} ms | {mark}")
        timed.append(result)

    manager.stop()
    print(f"  Cooling down {cooldown}s...")
    time.sleep(cooldown)

    ok = [r["latency_ms"] for r in timed if r["status"] == "success"]
    minimum = min(ok) if ok else None
    average = sum(ok) / len(ok) if ok else None
    print(f"  Min: {minimum:.1f} ms" if minimum is not None else "  Min: N/A")
    return {
        "phase": name,
        "phase_config": {k: v for k, v in phase.items() if k != "name"},
        "warmup": warmup,
        "timed_results": timed,
        "min_latency_ms": round(minimum, 1) if minimum is not None else None,
        "avg_latency_ms": round(average, 1) if average is not None else None,
        "min_latency_s": round(minimum / 1000.0, 3) if minimum is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default="morphingdb,hermes")
    parser.add_argument("--phases", default="low,medium,high")
    parser.add_argument("--workload", choices=("year_predict", "cifar10"), default="year_predict")
    parser.add_argument("--row-count", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=TIMED_RUNS)
    parser.add_argument("--cooldown", type=int, default=COOLDOWN_SECONDS)
    args = parser.parse_args()
    workload = get_workload(args.workload)
    row_count = args.row_count if args.row_count is not None else workload["row_count"]

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = [s for s in systems if s not in RUNNERS]
    if unknown:
        raise SystemExit(f"unknown systems {unknown}; choices: {sorted(RUNNERS)}")
    wanted = [p.strip() for p in args.phases.split(",") if p.strip()]
    known = {phase["name"]: phase for phase in workload["phases"]}
    missing = [p for p in wanted if p not in known]
    if missing:
        raise SystemExit(f"unknown phases {missing}; choices: {sorted(known)}")
    hermes_script = HERMES_SCRIPTS[args.workload]
    if not hermes_script.exists():
        raise SystemExit(f"missing {hermes_script}")

    print(f"Baseline comparison under the ablation {workload['kind']} load")
    print(f"  dataset:   {args.workload} n={row_count}")
    print(f"  systems:   {', '.join(systems)}")
    print(f"  phases:    {', '.join(wanted)}")
    print(f"  timed:     {args.repetitions} (report minimum)")

    for system in systems:
        started = time.time()
        phases = [
            run_phase(system, args.workload, known[name], row_count, args.cooldown, args.repetitions)
            for name in wanted
        ]
        payload = {
            "timestamp": datetime.now().isoformat(),
            "experiment": "optimization_ablation_baselines",
            "system": system,
            "workload": args.workload,
            "config": {
                "row_count": row_count,
                "device": workload["default_device_when_off"],
                "warmup_runs": WARMUP_RUNS,
                "timed_runs": args.repetitions,
                "cooldown_seconds": args.cooldown,
                "load": (
                    "spin_load pinned to CPU 0, low=1 medium=4 high=12"
                    if args.workload == "year_predict"
                    else "matmul on GPU 0, low=2x2GB medium=3x2GB high=8x1.5GB"
                ),
                "reported_stat": "minimum of successful timed runs",
            },
            "total_time_seconds": round(time.time() - started, 1),
            "phases": phases,
        }
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"{args.workload}_{system}_{stamp}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print(f"\n{system}  {args.workload} n={row_count}")
        for phase in phases:
            seconds = f"{phase['min_latency_s']:.3f}" if phase["min_latency_s"] is not None else "N/A"
            print(f"  {phase['phase']:<8s} {seconds:>8s} s")
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
