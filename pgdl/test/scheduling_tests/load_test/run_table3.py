#!/usr/bin/env python3
"""Table 3 of the paper: query latency under resource contention.

Reproduces paper_workspace/MDB_Agent/tables/scheduling_load.tex. Nine cells,
three systems by three workloads. The reported value of a cell is the minimum
of the timed runs after one discarded warmup.

    workload      n      load                              systems
    year_predict  10000  CPU overcommit, 20/60/120 on 0-19 DBAgent, MorphingDB, Hermes
    cifar10       1000   GPU matmul on GPU 0               DBAgent, MorphingDB, Hermes
    hotpotqa      20     GPU matmul on GPU 0               DBAgent, MorphingDB, Hermes

Hermes is reported as generation+execution. Generation is a fixed cost measured
once with no load (or passed in with --generation-ms); only execution runs
under load. The published generation costs, used when generation is skipped:

    year_predict  24996.5 ms
    cifar10      227541.3 ms
    hotpotqa     208110.0 ms

DBAgent and MorphingDB time the whole query, from connect to fetch. Hermes
times data loading plus inference, with the model loaded outside the timer.

The two flags in pgdl/src/pgdl/model_agent.cpp must be set before this runs:
ADD_OPTIMIZATION_LOGIC true, ALWAYS_TRAINING false, then rebuild and install
pgdl.so. Do not run two cells at once; every cell pins CPUs 0-19.

    python3 run_table3.py --list
    python3 run_table3.py --cell dbagent_year_predict --cooldown 60
    python3 run_table3.py --workload year_predict
    python3 run_table3.py --skip-generation
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import psycopg2

from scheduling_tests.load_test.harness import (
    ALLOWED_CPU_MASK,
    CPU_PHASES,
    DB_CONFIG,
    GPU_PHASES,
    CpuLoad,
    CpuMonitor,
    GpuLoad,
    cooldown,
    gpu_util,
    pin_backend,
    pin_this_process,
    query_status,
)
from support.test_result_paths import SCHEDULING_RESULTS_DIR, ensure_dir

HERE = os.path.dirname(os.path.abspath(__file__))
EXEC_DIR = os.path.join(HERE, "baseline_exec")
RESULTS_DIR = ensure_dir(SCHEDULING_RESULTS_DIR / "load_test")

HERMES_CLI = "/home/why/.local/bin/hermes"
MODEL_DIR = os.path.join(HERE, "..", "..", "morphingdb_test", "models")
TOOLS_DIR = os.path.join(HERE, "..", "..", "tools")
YEAR_MODEL = "/home/why/baseline/models/year_predict.pt"
YEAR_DATA = "/home/why/baseline/data/series/yead_predict/YearPredictionMSD.csv"
CIFAR_MODEL = "/home/why/baseline/models/googlenet_cifar10.pt"

PUBLISHED_GENERATION_MS = {
    "year_predict": 24996.5,
    "cifar10": 227541.3,
    "hotpotqa": 208110.0,
}

PAPER_SECONDS = {
    ("dbagent", "year_predict"): (0.764, 1.092, 3.504),
    ("morphingdb", "year_predict"): (1.079, 1.892, 4.918),
    ("hermes", "year_predict"): (0.233, 0.239, 0.982),
    ("dbagent", "cifar10"): (28.2, 31.3, 39.7),
    ("morphingdb", "cifar10"): (31.8, 35.6, 48.0),
    ("hermes", "cifar10"): (62.9, 93.9, 200.6),
    ("dbagent", "hotpotqa"): (92.0, 149.9, 272.5),
    ("morphingdb", "hotpotqa"): (168.6, 268.7, 345.1),
    ("hermes", "hotpotqa"): (148.9, 298.3, 610.0),
}

WARMUP_RUNS = 0
STATEMENT_TIMEOUT = "1800s"

QUERIES = {
    "dbagent_year_predict": (
        "SELECT unnest(db_agent_single('series', sub_table.data)) AS score "
        "FROM (SELECT * FROM year_predict_test LIMIT {n}) AS sub_table;"
    ),
    "morphingdb_year_predict": (
        "SELECT predict_batch_float8('year_predict', 'cpu', data) "
        "OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) "
        "FROM year_predict_test LIMIT {n};"
    ),
    "dbagent_cifar10": (
        "SELECT unnest(db_agent_single('image_classification', sub_table.image_vector)) AS score "
        "FROM (SELECT * FROM cifar_image_vector_table LIMIT {n}) AS sub_table;"
    ),
    "morphingdb_cifar10": (
        "SELECT predict_batch_float8('googlenet_cifar10', 'gpu:0', image_vector) "
        "OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) "
        "FROM cifar_image_vector_table LIMIT {n};"
    ),
    "dbagent_hotpotqa": (
        "SELECT count(*) FROM ("
        "SELECT unnest(db_agent_single('reasoning', sub_table.reasoning_vec)) AS score "
        "FROM (SELECT * FROM reasoning_vector_test LIMIT {n}) AS sub_table) t;"
    ),
    "morphingdb_hotpotqa_cross": (
        "SELECT predict_batch_float8('cross_encoder', 'gpu:0', reasoning_vec) "
        "OVER (ROWS BETWEEN CURRENT ROW AND 15 FOLLOWING) "
        "FROM reasoning_vector_test LIMIT {n};"
    ),
}

GENERATION_PROMPTS = {
    "year_predict": (
        f"Load the model at {YEAR_MODEL} using torch.jit.load on CPU. "
        f"Read the first {{n}} rows of {YEAR_DATA} with pandas, drop the value90 column, "
        "and run the model in batches of 32. Report the inference time."
    ),
    "cifar10": (
        f"Load the model at {CIFAR_MODEL} using torch.jit.load on GPU (device='cuda:0'). "
        "Fetch {n} image vectors from the table cifar_image_vector_table in a local PostgreSQL "
        "database (host=localhost, user=why, password=123456, dbname=postgres). "
        "Each image_vector is a mvec type that represents a preprocessed 1x3x224x224 float32 tensor. "
        "Run the model in batches of 32 and report the inference time."
    ),
    "hotpotqa": (
        f"Load cross_encoder.pt from {MODEL_DIR} using torch.jit.load on GPU (device='cuda:0') "
        "and deberta_reader.pt on CPU. "
        "Fetch reasoning vectors from reasoning_vector_test and reasoning_vector_step2 from PostgreSQL "
        "(host=localhost, user=why, password=123456, dbname=postgres). "
        "Each reasoning_vec is a mvec (1x4x128 float32). "
        "Score all {n} samples with cross_encoder in batch. "
        "Run deberta_reader per-sample on CPU to extract span logits. "
        f"Call {TOOLS_DIR}/reader_decoder.py via subprocess per sample with --ids argument. "
        "Report precise timing for each stage."
    ),
}

EXEC_SCRIPTS = {
    "year_predict": os.path.join(EXEC_DIR, "hermes_year_predict_exec.py"),
    "cifar10": os.path.join(EXEC_DIR, "hermes_cifar10_exec.py"),
    "hotpotqa": os.path.join(EXEC_DIR, "hermes_reasoning_exec.py"),
}

CELLS = [
    {"name": "dbagent_year_predict", "system": "dbagent", "workload": "year_predict",
     "n": 10000, "load": "cpu", "runs": 3},
    {"name": "morphingdb_year_predict", "system": "morphingdb", "workload": "year_predict",
     "n": 10000, "load": "cpu", "runs": 3},
    {"name": "hermes_year_predict", "system": "hermes", "workload": "year_predict",
     "n": 10000, "load": "cpu", "runs": 3},
    {"name": "dbagent_cifar10", "system": "dbagent", "workload": "cifar10",
     "n": 1000, "load": "gpu", "runs": 3},
    {"name": "morphingdb_cifar10", "system": "morphingdb", "workload": "cifar10",
     "n": 1000, "load": "gpu", "runs": 3},
    {"name": "hermes_cifar10", "system": "hermes", "workload": "cifar10",
     "n": 1000, "load": "cpu", "runs": 3},
    {"name": "dbagent_hotpotqa", "system": "dbagent", "workload": "hotpotqa",
     "n": 20, "load": "gpu", "runs": 1},
    {"name": "morphingdb_hotpotqa", "system": "morphingdb", "workload": "hotpotqa",
     "n": 20, "load": "gpu", "runs": 1},
    {"name": "hermes_hotpotqa", "system": "hermes", "workload": "hotpotqa",
     "n": 20, "load": "gpu", "runs": 1},
]
CELL_BY_NAME = {cell["name"]: cell for cell in CELLS}


def connect():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    pin_backend(cur)
    cur.execute("SELECT register_process()")
    cur.execute(f"SET statement_timeout = '{STATEMENT_TIMEOUT}'")
    return conn, cur


def run_sql(sql, reuse=None):
    """Time one statement. reuse is an open (conn, cur); otherwise connect per call."""
    own_connection = reuse is None
    start = time.perf_counter()
    try:
        if own_connection:
            conn, cur = connect()
        else:
            conn, cur = reuse
        cur.execute(sql)
        rows = cur.fetchall()
        if own_connection:
            conn.close()
        return {"latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "status": "success", "rows_returned": len(rows)}
    except Exception as exc:
        return {"latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "status": query_status(exc), "rows_returned": 0, "error": str(exc)}


def prepare_hotpotqa(n):
    """Step-2 vectors and the CPU deberta reader, both outside the timer."""
    import numpy as np
    import torch

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "SELECT get_mvec_data(text_vec) FROM reasoning_vector_step2 ORDER BY id LIMIT %s",
        (n * 2,),
    )
    vecs = [np.array(row[0], dtype=np.float32).reshape(4, 128) for row in cur.fetchall()]
    conn.close()

    tensors = []
    for i in range(n):
        stacked = np.stack(
            [vecs[i * 2][:2, :], vecs[i * 2 + 1][:2, :]], axis=0,
        ).astype(np.float32)
        tensors.append(torch.from_numpy(stacked))

    reader = torch.jit.load(os.path.join(MODEL_DIR, "deberta_reader.pt"), map_location="cpu")
    reader.eval()
    return vecs, tensors, reader


def run_morphingdb_hotpotqa(n, vecs, tensors, reader):
    """cross_encoder through MorphingDB, then local deberta and reader_decoder."""
    start = time.perf_counter()
    try:
        conn, cur = connect()
        cur.execute(QUERIES["morphingdb_hotpotqa_cross"].format(n=n))
        cur.fetchall()
        conn.close()
    except Exception as exc:
        return {"latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "status": query_status(exc), "error": str(exc)}

    import torch

    span_parts = []
    try:
        for i in range(n):
            with torch.no_grad():
                output = reader(tensors[i])
            start_logits, end_logits = output[:, 0, :], output[:, 1, :]
            for j in (0, 1):
                input_ids = vecs[i * 2 + j][0, :].astype(int).tolist()
                start_idx = int(start_logits[j].argmax().item())
                end_idx = int(end_logits[j].argmax().item())
                span_parts.append(",".join(str(token) for token in input_ids[start_idx:end_idx + 1]))
    except Exception as exc:
        return {"latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "status": "error", "error": str(exc)}

    decoder = os.path.join(TOOLS_DIR, "reader_decoder.py")
    try:
        for i in range(n):
            span = f"{span_parts[i * 2]};{span_parts[i * 2 + 1]}"
            subprocess.run(
                ["uv", "run", "--project", os.path.join(TOOLS_DIR, ".."), "python", decoder, "--ids", span],
                capture_output=True, text=True, cwd=TOOLS_DIR, timeout=120,
            )
    except Exception as exc:
        return {"latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "status": "error", "error": str(exc)}

    return {"latency_ms": round((time.perf_counter() - start) * 1000, 1), "status": "success"}


def run_hermes_exec(workload, n, timeout):
    # HotpotQA's reader_decoder must run under the project venv. A bare
    # python3 resolves to miniconda, whose transformers build roughly doubles
    # the decode and leaves control bytes as literals.
    interpreter = os.path.join(TOOLS_DIR, "..", ".venv", "bin", "python3") if workload == "hotpotqa" else sys.executable
    cmd = ["taskset", "-c", ALLOWED_CPU_MASK, interpreter, EXEC_SCRIPTS[workload], str(n)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""} if workload == "year_predict" else None
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "latency_ms": timeout * 1000}
    if result.returncode != 0:
        return {"status": "error", "latency_ms": 0, "error": result.stderr[:300]}
    line = next((row.strip() for row in reversed(result.stdout.splitlines())
                 if row.strip().startswith("{")), None)
    if line is None:
        return {"status": "error", "latency_ms": 0, "error": "no JSON output"}
    data = json.loads(line)
    return {"status": "success", "latency_ms": data["total_ms"],
            "inference_ms": data.get("inference_ms")}


def measure_generation(workload, n, runs, timeout):
    prompt = GENERATION_PROMPTS[workload].format(n=n)
    print(f"\n{'=' * 70}")
    print(f"  GENERATION (no load, {runs} runs, min kept)")
    print(f"{'=' * 70}")
    times = []
    for i in range(runs):
        start = time.perf_counter()
        try:
            result = subprocess.run(
                [HERMES_CLI, "-z", prompt], capture_output=True, text=True, timeout=timeout,
            )
            elapsed = (time.perf_counter() - start) * 1000
            status = "success" if result.returncode == 0 else "error"
        except subprocess.TimeoutExpired:
            elapsed, status = timeout * 1000, "timeout"
        print(f"  Run {i + 1}: {elapsed:.0f} ms [{status}]")
        if status == "success":
            times.append(elapsed)
    if not times:
        fallback = PUBLISHED_GENERATION_MS[workload]
        print(f"  all generation runs failed, using the published {fallback:.1f} ms")
        return fallback, []
    chosen = min(times)
    print(f"  Min generation time: {chosen:.1f} ms")
    return round(chosen, 1), [round(t, 1) for t in times]


def start_load(cell):
    if cell["load"] == "cpu":
        manager = CpuLoad()
        return manager, CPU_PHASES, 3, CpuMonitor()
    manager = GpuLoad()
    return manager, GPU_PHASES, 8, None


def begin_phase(manager, phase):
    if "mem_gb" in phase:
        manager.start(phase["n_workers"], phase["mem_gb"], phase["oom_style"])
        label = "oom-style" if phase["oom_style"] else "new-style"
        print(f"  started {phase['n_workers']} GPU workers ({label}, {phase['mem_gb']} GB each)")
    else:
        manager.start(phase["n_workers"])
        print(f"  started {phase['n_workers']} spin workers on CPUs 0-19 "
              f"({phase['n_workers'] / 20:.0f}:1)")


def sample_load(monitor):
    if monitor is None:
        util, _ = gpu_util()
        return util
    return monitor.get_usage()


def run_cell(cell, args, generation_ms):
    n = args.rows if args.rows is not None else cell["n"]
    runs = args.repetitions if args.repetitions is not None else cell["runs"]
    manager, phases, settle_s, monitor = start_load(cell)
    prepared = None
    if cell["name"] == "morphingdb_hotpotqa":
        print("  loading deberta_reader and step-2 vectors (outside the timer)...")
        prepared = prepare_hotpotqa(n)

    print(f"\n{'#' * 70}")
    print(f"  {cell['name']}   n={n}   {cell['load']} load   "
          f"{WARMUP_RUNS} warmup + {runs} timed, report min")
    print(f"{'#' * 70}")

    phase_results = []
    for phase in phases:
        print(f"\n{'=' * 70}")
        print(f"  PHASE {phase['name'].upper()}")
        print(f"{'=' * 70}")
        begin_phase(manager, phase)
        print(f"  settling for {settle_s}s...")
        time.sleep(settle_s)
        if monitor is not None:
            monitor.warmup()
            time.sleep(0.2)
        observed = sample_load(monitor)
        print(f"  load before queries: {observed:.1f}%")

        def once():
            if cell["system"] == "hermes":
                timeout = 900 if cell["workload"] == "hotpotqa" else 300
                return run_hermes_exec(cell["workload"], n, timeout)
            if cell["name"] == "morphingdb_hotpotqa":
                return run_morphingdb_hotpotqa(n, *prepared)
            sql = QUERIES[cell["name"]].format(n=n)
            if cell["load"] == "gpu" and cell["system"] == "dbagent":
                return run_sql(sql, once.connection)
            return run_sql(sql)

        once.connection = connect() if cell["load"] == "gpu" and cell["system"] == "dbagent" else None

        print("\n  warmup (discarded)")
        warmup = once()
        print(f"    {warmup['latency_ms']:.1f} ms [{warmup['status']}]")

        print(f"\n  timed runs ({runs})")
        timed = []
        samples = []
        for i in range(runs):
            before = sample_load(monitor)
            result = once()
            after = sample_load(monitor)
            samples.append((before + after) / 2)
            mark = "OK" if result["status"] == "success" else "!!"
            print(f"    {i + 1}: {result['latency_ms']:10.1f} ms  {mark}")
            timed.append(result)
        if once.connection is not None:
            once.connection[0].close()
        manager.stop()

        ok = [r["latency_ms"] for r in timed if r["status"] == "success"]
        minimum = min(ok) if ok else None
        average = sum(ok) / len(ok) if ok else None
        record = {
            "phase": phase["name"],
            "n_workers": phase["n_workers"],
            "load_pct": round(sum(samples) / len(samples), 1) if samples else None,
            "warmup": warmup,
            "timed_results": timed,
            "min_latency_ms": round(minimum, 1) if minimum is not None else None,
            "avg_latency_ms": round(average, 1) if average is not None else None,
        }
        if generation_ms is not None and minimum is not None:
            record["min_total_ms"] = round(generation_ms + minimum, 1)
        phase_results.append(record)
        shown = f"{minimum:.1f} ms" if minimum is not None else "failed"
        print(f"\n  min of {phase['name']}: {shown}", flush=True)
        partial = os.path.join(RESULTS_DIR, "table3_partial.jsonl")
        with open(partial, "a") as handle:
            handle.write(json.dumps({
                "cell": cell["name"], "n": n, "phase": phase["name"],
                "min_latency_ms": record["min_latency_ms"],
                "load_pct": record["load_pct"],
            }) + "\n")

        if phase is not phases[-1]:
            cooldown(args.cooldown, monitor)

    return {
        "cell": cell["name"],
        "system": cell["system"],
        "workload": cell["workload"],
        "load": cell["load"],
        "n": n,
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": runs,
        "generation_ms": generation_ms,
        "phases": phase_results,
    }


def paper_line(record):
    """One comparison line: measured seconds against the published cell."""
    key = (record["system"], record["workload"])
    published = PAPER_SECONDS[key]
    names = ["low", "medium", "high"]
    measured = []
    for phase in record["phases"]:
        value = phase["min_latency_ms"]
        measured.append(None if value is None else value / 1000.0)
    cells = []
    for got, expected in zip(measured, published):
        if got is None:
            cells.append("fail")
            continue
        gap = (got - expected) / expected * 100
        cells.append(f"{got:.3f} vs {expected:.3f} ({gap:+.1f}%)")
    gen = ""
    if record["generation_ms"] is not None:
        gen = f"   gen {record['generation_ms'] / 1000:.3f}s + exec"
    return f"  {record['cell']:<24}{gen}\n    " + "   ".join(
        f"{name}: {cell}" for name, cell in zip(names, cells)
    )


def select_cells(args):
    if args.cell:
        unknown = [name for name in args.cell if name not in CELL_BY_NAME]
        if unknown:
            raise SystemExit(f"unknown cell: {', '.join(unknown)}; see --list")
        return [CELL_BY_NAME[name] for name in args.cell]
    if args.workload:
        chosen = [cell for cell in CELLS if cell["workload"] in args.workload]
        if not chosen:
            raise SystemExit(f"no cells for workload: {', '.join(args.workload)}")
        return chosen
    return list(CELLS)


def main():
    parser = argparse.ArgumentParser(description="Table 3 load experiment, one entry point")
    parser.add_argument("--list", action="store_true", help="print the nine cells and exit")
    parser.add_argument("--cell", nargs="+", help="run only these cells")
    parser.add_argument("--workload", nargs="+", choices=["year_predict", "cifar10", "hotpotqa"])
    parser.add_argument("--rows", type=int, help="override the row count of every selected cell")
    parser.add_argument("--repetitions", type=int, help="override timed runs per phase")
    parser.add_argument("--cooldown", type=int, default=60)
    parser.add_argument("--skip-generation", action="store_true",
                        help="use the published Hermes generation costs instead of calling hermes")
    parser.add_argument("--generation-ms", type=float,
                        help="one generation cost for every selected Hermes cell")
    parser.add_argument("--generation-runs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = parser.parse_args()

    if args.list:
        print(f"{'cell':<24}{'system':<12}{'workload':<14}{'n':>6}  {'load':<5}{'runs':>5}")
        for cell in CELLS:
            print(f"{cell['name']:<24}{cell['system']:<12}{cell['workload']:<14}"
                  f"{cell['n']:>6}  {cell['load']:<5}{cell['runs']:>5}")
        return 0

    chosen = select_cells(args)
    if args.dry_run:
        for cell in chosen:
            extra = ""
            if cell["system"] == "hermes":
                if args.generation_ms is not None:
                    extra = f"  generation: given {args.generation_ms:.1f} ms"
                elif args.skip_generation:
                    extra = f"  generation: published {PUBLISHED_GENERATION_MS[cell['workload']]:.1f} ms"
                else:
                    extra = f"  generation: {args.generation_runs} hermes runs, keep min"
            print(f"  {cell['name']:<24} n={cell['n']:<6} {cell['load']} load, "
                  f"{cell['runs']} timed runs{extra}")
        print("  dry run, nothing started")
        return 0

    pin_this_process()
    generation = {}
    records = []
    started = time.time()
    for cell in chosen:
        gen_ms = None
        if cell["system"] == "hermes":
            if args.generation_ms is not None:
                gen_ms = args.generation_ms
            elif args.skip_generation:
                gen_ms = PUBLISHED_GENERATION_MS[cell["workload"]]
                print(f"\n  {cell['name']}: published generation cost {gen_ms:.1f} ms")
            elif cell["workload"] not in generation:
                n = args.rows if args.rows is not None else cell["n"]
                timeout = 900 if cell["workload"] == "hotpotqa" else 600
                gen_ms, _ = measure_generation(cell["workload"], n, args.generation_runs, timeout)
                generation[cell["workload"]] = gen_ms
            else:
                gen_ms = generation[cell["workload"]]
        records.append(run_cell(cell, args, gen_ms))

    print(f"\n\n{'=' * 70}")
    print("  MEASURED vs PUBLISHED (execution seconds, min of timed runs)")
    print(f"{'=' * 70}")
    for record in records:
        print(paper_line(record))
    print(f"\n  elapsed {((time.time() - started) / 60):.1f} min")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": datetime.now().isoformat(),
        "table": "paper_workspace/MDB_Agent/tables/scheduling_load.tex",
        "paper_seconds": {f"{k[0]}_{k[1]}": v for k, v in PAPER_SECONDS.items()},
        "elapsed_seconds": round(time.time() - started, 1),
        "cells": records,
    }
    path = os.path.join(RESULTS_DIR, f"table3_{stamp}.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"  saved {path}")
    failed = [r["cell"] for r in records if any(p["min_latency_ms"] is None for p in r["phases"])]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
