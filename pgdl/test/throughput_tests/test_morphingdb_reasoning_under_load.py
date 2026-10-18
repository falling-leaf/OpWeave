#!/usr/bin/env python3
"""
MorphingDB HotpotQA (reasoning) Query Performance Under GPU Contention.

Pipeline (same models as DBAgent, all steps timed):
  Step 1: predict_batch_float8('cross_encoder', 'gpu:0', reasoning_vec) via DB
  Step 2: deberta_reader.pt locally (CPU, per-sample, same model as DBAgent)
  Step 3: reader_decoder.py subprocess per sample (same as DBAgent)

Total latency is the sum of all three steps; this baseline should exceed DBAgent
because MorphingDB has no intelligent scheduling, no session reuse, and no
persistent model cache between Python-side deberta calls.

GPU load phases (matmul workers on GPU 0, identical to DBAgent test):
  - Low:    2 workers x 2GB  new-style (no sync, 50ms sleep)
  - Medium: 3 workers x 2GB  new-style
  - High:   8 workers x 1.5GB oom-style (sync + per-iter alloc, 1ms sleep)

Benchmark: 1 warmup + 1 timed run, report minimum.

Usage:
    python test_morphingdb_reasoning_under_load.py
    python test_morphingdb_reasoning_under_load.py --row-count 20 --repetitions 1

Originally one of the nine Table 3 cell scripts (kept under
recycle/scheduling_tests_table3/ after run_table3.py replaced them). It lives
here because test_reasoning_baseline_concurrency.py imports its
load_step2_vecs / build_step2_tensors / run_morphingdb_query as the
MorphingDB baseline runner.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import json
import subprocess
import subprocess as sp
import argparse
from datetime import datetime
from support.test_result_paths import SCHEDULING_RESULTS_DIR, ensure_dir

import numpy as np
import torch
import psycopg2

ALLOWED_CPUS = list(range(20))
ALLOWED_CPU_MASK = ",".join(str(c) for c in ALLOWED_CPUS)

RESULTS_DIR = ensure_dir(SCHEDULING_RESULTS_DIR)

DB_CONFIG = {
    "host": "localhost", "port": "5432", "user": "why",
    "dbname": "postgres", "password": "123456",
}

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + "/pgdl/test",
    "morphingdb_test", "models")
TOOLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + "/pgdl/test",
    "tools")

# cross_encoder batch window query (same as DBAgent's batch GPU call)
CROSS_QUERY = (
    "SELECT predict_batch_float8('cross_encoder', 'gpu:0', reasoning_vec) "
    "OVER (ROWS BETWEEN CURRENT ROW AND 15 FOLLOWING) "
    "FROM reasoning_vector_test LIMIT {row_count};"
)

GPU_LOAD_PHASES = [
    {"name": "low",    "n_workers": 2, "mem_gb": 2.0,  "oom_style": False},
    {"name": "medium", "n_workers": 3, "mem_gb": 2.0,  "oom_style": False},
    {"name": "high",   "n_workers": 8, "mem_gb": 1.5,  "oom_style": True},
]

COOLDOWN_SECONDS = 60
STABILIZATION_SECONDS = 8
WARMUP_RUNS = 1
TIMED_RUNS = 3

NEW_WORKER_SCRIPT = """\
import torch, time, sys, os
os.sched_setaffinity(0, [0])
mem_gb = float(sys.argv[1]); stop_file = sys.argv[2]
n = min(int((mem_gb * 1024**3 / 4) ** 0.5), 8192)
a = torch.randn(n, n, device='cuda:0', dtype=torch.float32)
b = torch.randn(n, n, device='cuda:0', dtype=torch.float32)
c = torch.zeros(n, n, device='cuda:0', dtype=torch.float32)
while not os.path.exists(stop_file):
    with torch.no_grad():
        c.copy_(torch.matmul(a, b))
    time.sleep(0.05)
"""

OOM_WORKER_SCRIPT = """\
import torch, time, sys, os
os.sched_setaffinity(0, [0])
mem_gb = float(sys.argv[1]); stop_file = sys.argv[2]
n = min(int((mem_gb * 1024**3 / 4) ** 0.5), 8192)
a = torch.randn(n, n, device='cuda:0', dtype=torch.float32)
b = torch.randn(n, n, device='cuda:0', dtype=torch.float32)
while not os.path.exists(stop_file):
    with torch.no_grad():
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    time.sleep(0.001)
"""


class GpuLoadManager:
    def __init__(self):
        self.procs = []
        self.stop_file = None
        self.script_file = None

    def start(self, n_workers, mem_gb, oom_style=False):
        import tempfile
        self.stop_file = tempfile.mktemp(suffix='.stop')
        sf = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        sf.write(OOM_WORKER_SCRIPT if oom_style else NEW_WORKER_SCRIPT)
        sf.flush()
        sf.close()
        self.script_file = sf.name

        self.procs = []
        for _ in range(n_workers):
            p = subprocess.Popen(
                ["taskset", "-c", ALLOWED_CPU_MASK,
                 sys.executable, self.script_file,
                 str(mem_gb), self.stop_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.procs.append(p)

    def stop(self):
        if self.stop_file:
            try:
                with open(self.stop_file, 'w') as f:
                    f.write('stop')
            except Exception:
                pass
        time.sleep(2)
        for p in self.procs:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        self.procs.clear()
        for f in [self.stop_file, self.script_file]:
            try:
                if f:
                    os.unlink(f)
            except Exception:
                pass


def get_gpu_util():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        parts = [p.strip() for p in r.stdout.strip().split(',')]
        return float(parts[0]), float(parts[1])
    except Exception:
        return 0.0, 0.0


def load_step2_vecs(row_count):
    """Fetch reasoning_vector_step2 from DB; returns list of (4,128) float32 arrays."""
    n = row_count * 2
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "SELECT get_mvec_data(text_vec) FROM reasoning_vector_step2 ORDER BY id LIMIT %s",
        (n,)
    )
    vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()
    return vecs


def build_step2_tensors(step2_vecs, row_count):
    """Stack pairs of step2 vecs into (2,2,128) tensors for deberta_reader."""
    tensors = []
    for i in range(row_count):
        idx = i * 2
        t = np.stack(
            [step2_vecs[idx][:2, :], step2_vecs[idx + 1][:2, :]],
            axis=0
        ).astype(np.float32)
        tensors.append(torch.from_numpy(t))
    return tensors


# Models pre-loaded once (excluded from per-run timing, same as hermes_reasoning_exec.py)
print("Pre-loading models (excluded from timing)...")
_model_reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location=torch.device("cpu"))
_model_reader.eval()

print("Pre-loading step2 vectors (warm cache)...")
_step2_vecs_cache = None  # loaded lazily on first run

print("Models ready.\n")


def run_morphingdb_query(row_count, step2_tensors, step2_vecs):
    """
    Full MorphingDB reasoning pipeline:
      1. predict_batch_float8('cross_encoder', 'gpu:0') via DB window query
      2. deberta_reader.pt locally on CPU, per sample
      3. reader_decoder.py subprocess per sample

    Models pre-loaded; data loading + inference + decode all timed.
    """
    start_total = time.perf_counter()

    # ---- Step 1: cross_encoder via MorphingDB window query (GPU) ----
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("SELECT pg_backend_pid()")
        pid = cur.fetchone()[0]
        subprocess.run(
            ["taskset", "-p", "-c", ALLOWED_CPU_MASK, str(pid)],
            capture_output=True, timeout=5
        )

        cur.execute("SELECT register_process()")
        cur.execute("SET statement_timeout = '1800s'")

        t1 = time.perf_counter()
        cur.execute(CROSS_QUERY.format(row_count=row_count))
        cur.fetchall()
        cross_ms = (time.perf_counter() - t1) * 1000
        conn.close()

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_total) * 1000
        err = str(e)
        status = "timeout" if "timeout" in err.lower() or "canceling" in err.lower() else "error"
        return {"latency_ms": round(elapsed_ms, 1), "status": status, "error": err}

    # ---- Step 2: deberta_reader locally (CPU, per-sample) ----
    deberta_ms = 0.0
    span_parts = []  # list of token-id span strings, 2 per sample
    try:
        for i in range(row_count):
            td = time.perf_counter()
            with torch.no_grad():
                reader_output = _model_reader(step2_tensors[i])
            start_logits = reader_output[:, 0, :]
            end_logits   = reader_output[:, 1, :]
            deberta_ms += (time.perf_counter() - td) * 1000

            for j in [0, 1]:
                vec_idx = i * 2 + j
                input_ids = step2_vecs[vec_idx][0, :].astype(int).tolist()
                s = int(start_logits[j].argmax().item())
                e = int(end_logits[j].argmax().item())
                span_parts.append(",".join(str(t) for t in input_ids[s:e + 1]))

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_total) * 1000
        return {"latency_ms": round(elapsed_ms, 1), "status": "error", "error": str(e)}

    # ---- Step 3: reader_decoder.py subprocess per sample ----
    decoder_ms = 0.0
    try:
        for i in range(row_count):
            res_str = f"{span_parts[i * 2]};{span_parts[i * 2 + 1]}"
            td3 = time.perf_counter()
            sp.run(
                ["python3", f"{TOOLS_DIR}/reader_decoder.py", "--ids", res_str],
                capture_output=True, text=True, cwd=TOOLS_DIR, timeout=120
            )
            decoder_ms += (time.perf_counter() - td3) * 1000

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_total) * 1000
        return {"latency_ms": round(elapsed_ms, 1), "status": "error", "error": str(e)}

    total_ms = (time.perf_counter() - start_total) * 1000
    return {
        "latency_ms": round(total_ms, 1),
        "cross_ms":   round(cross_ms, 1),
        "deberta_ms": round(deberta_ms, 1),
        "decoder_ms": round(decoder_ms, 1),
        "status": "success",
    }


def run_phase(phase, row_count, step2_tensors, step2_vecs, cooldown, timed_runs=TIMED_RUNS):
    name      = phase["name"]
    n_workers = phase["n_workers"]
    mem_gb    = phase["mem_gb"]
    oom_style = phase.get("oom_style", False)

    print(f"\n{'='*70}")
    print(f"  PHASE: {name.upper()} ({n_workers} workers x {mem_gb}GB = {n_workers*mem_gb:.0f}GB)")
    print(f"{'='*70}")

    manager = GpuLoadManager()
    manager.start(n_workers, mem_gb, oom_style=oom_style)
    style_label = "oom-style" if oom_style else "new-style"
    print(f"  Started {n_workers} GPU load workers [{style_label}]")
    print(f"  Stabilizing for {STABILIZATION_SECONDS}s...")
    time.sleep(STABILIZATION_SECONDS)

    util, mem_used = get_gpu_util()
    print(f"  GPU 0 util: {util:.0f}%  mem_used: {mem_used:.0f}MB")

    print(f"\n  Warmup run (discarded)...")
    wr = run_morphingdb_query(row_count, step2_tensors, step2_vecs)
    cross_w  = wr.get('cross_ms', 0)
    deb_w    = wr.get('deberta_ms', 0)
    dec_w    = wr.get('decoder_ms', 0)
    print(f"    Warmup: {wr['latency_ms']:.1f}ms "
          f"(cross={cross_w:.1f} deberta={deb_w:.1f} decoder={dec_w:.1f}) [{wr['status']}]")

    print(f"\n  Timed runs ({timed_runs})...")
    print(f"  {'Run':>4s} | {'Total (ms)':>10s} | {'Cross (ms)':>10s} | "
          f"{'Deberta (ms)':>12s} | {'Decoder (ms)':>12s} | {'GPU%':>5s} | Status")
    print(f"  {'-'*4} | {'-'*10} | {'-'*10} | {'-'*12} | {'-'*12} | {'-'*5} | {'-'*8}")

    timed_results = []
    gpu_utils = []
    for i in range(timed_runs):
        util_before, _ = get_gpu_util()
        r = run_morphingdb_query(row_count, step2_tensors, step2_vecs)
        util_after, _ = get_gpu_util()
        gpu_utils.append((util_before + util_after) / 2)

        sym = "OK" if r["status"] == "success" else "!!"
        print(f"  {i+1:4d} | {r['latency_ms']:10.1f} | {r.get('cross_ms',0):10.1f} | "
              f"{r.get('deberta_ms',0):12.1f} | {r.get('decoder_ms',0):12.1f} | "
              f"{gpu_utils[-1]:4.0f}% | {sym}")
        timed_results.append(r)

    manager.stop()

    print(f"\n  Cooling down for {cooldown}s...")
    cooldown_start = time.time()
    while time.time() - cooldown_start < cooldown:
        remaining = cooldown - (time.time() - cooldown_start)
        util_idle, _ = get_gpu_util()
        mins, secs = divmod(int(remaining), 60)
        print(f"    {mins:02d}:{secs:02d} remaining - GPU util: {util_idle:.0f}%")
        time.sleep(10)

    successful = [r["latency_ms"] for r in timed_results if r["status"] == "success"]
    min_lat = min(successful) if successful else None
    avg_lat = sum(successful) / len(successful) if successful else None
    avg_gpu = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0

    print(f"\n  Phase Summary:")
    print(f"    Min latency: {min_lat:.1f} ms" if min_lat else "    Min latency: N/A")
    print(f"    Avg GPU util: {avg_gpu:.1f}%")

    return {
        "phase": name,
        "n_workers": n_workers,
        "mem_gb_per_worker": mem_gb,
        "initial_gpu_util_pct": round(util, 1),
        "avg_gpu_util_during_queries": round(avg_gpu, 1),
        "warmup": wr,
        "timed_results": timed_results,
        "min_latency_ms": round(min_lat, 1) if min_lat is not None else None,
        "avg_latency_ms": round(avg_lat, 1) if avg_lat is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="MorphingDB reasoning latency under GPU load")
    parser.add_argument("--row-count", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=TIMED_RUNS)
    parser.add_argument("--cooldown", type=int, default=COOLDOWN_SECONDS)
    parser.add_argument("--save", action="store_true", default=True)
    args = parser.parse_args()

    os.sched_setaffinity(0, ALLOWED_CPUS)

    print("MorphingDB HotpotQA (reasoning) Under GPU Contention")
    print(f"  Pipeline:    cross_encoder(GPU via predict_batch_float8) + "
          f"deberta_reader(CPU local) + reader_decoder(subprocess x{args.row_count})")
    print(f"  Row count:   {args.row_count}")
    print(f"  Warmup runs: {WARMUP_RUNS} (discarded)")
    print(f"  Timed runs:  {args.repetitions} (report minimum)")
    print(f"  Cooldown:    {args.cooldown}s between phases")
    print(f"  Load type:   GPU matmul workers on GPU 0")

    print("\nLoading step2 vectors from DB...")
    step2_vecs    = load_step2_vecs(args.row_count)
    step2_tensors = build_step2_tensors(step2_vecs, args.row_count)
    print(f"  Loaded {len(step2_vecs)} step2 vectors, built {len(step2_tensors)} deberta input tensors")

    all_results = []
    overall_start = time.time()

    for phase in GPU_LOAD_PHASES:
        result = run_phase(phase, args.row_count, step2_tensors, step2_vecs,
                           args.cooldown, timed_runs=args.repetitions)
        all_results.append(result)

    total_time = time.time() - overall_start

    print(f"\n\n{'='*70}")
    print(f"  FINAL COMPARISON (n={args.row_count}, reasoning, MorphingDB full pipeline)")
    print(f"{'='*70}")
    print(f"  {'Phase':<8s} | {'Workers':>7s} | {'GPU%':>5s} | {'Min Lat (ms)':>12s} | "
          f"{'Avg Lat (ms)':>12s} | {'Slowdown':>8s}")
    print(f"  {'-'*8} | {'-'*7} | {'-'*5} | {'-'*12} | {'-'*12} | {'-'*8}")

    baseline_min = all_results[0]["min_latency_ms"] or 1
    for r in all_results:
        workers_str = f"{r['n_workers']}x{r['mem_gb_per_worker']}G"
        min_s = f"{r['min_latency_ms']:.1f}" if r["min_latency_ms"] is not None else "N/A"
        avg_s = f"{r['avg_latency_ms']:.1f}" if r["avg_latency_ms"] is not None else "N/A"
        slowdown = f"{r['min_latency_ms']/baseline_min:.2f}x" if r["min_latency_ms"] else "N/A"
        print(f"  {r['phase']:<8s} | {workers_str:>7s} | {r['avg_gpu_util_during_queries']:4.0f}% | "
              f"{min_s:>12s} | {avg_s:>12s} | {slowdown:>8s}")

    print(f"\n  Total time: {total_time/60:.1f} minutes")

    if args.save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "dataset": "reasoning",
                "row_count": args.row_count,
                "warmup_runs": WARMUP_RUNS,
                "timed_runs": args.repetitions,
                "cooldown_seconds": args.cooldown,
                "cross_query": CROSS_QUERY.format(row_count=args.row_count),
                "pipeline": (
                    "cross_encoder(GPU via predict_batch_float8) + "
                    "deberta_reader(CPU local) + reader_decoder(subprocess)"
                ),
                "load_type": "gpu_matmul_workers",
            },
            "total_time_seconds": round(total_time, 1),
            "phases": all_results,
        }
        filepath = os.path.join(RESULTS_DIR, f"morphingdb_reasoning_under_load_{ts}.json")
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to: {filepath}")


if __name__ == "__main__":
    main()
