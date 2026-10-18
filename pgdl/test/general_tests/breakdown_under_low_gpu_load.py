#!/usr/bin/env python3
"""One-shot HotpotQA breakdown under the scheduling low-GPU-load setting.

Load matches scheduling low phase:
  2 matmul workers x 2GB on GPU 0, new-style (no sync, 50ms sleep),
  workers pinned to CPUs 0-19, 8s stabilization.

Query and timing match the breakdown table source:
  db_agent_single('reasoning', ...) for n=20, per-agent times from
  /tmp/pgdl_agent_timing.log.

1 warmup (discarded) + 1 timed run, same as the scheduling HotpotQA run.
"""
import os
import re
import sys
import json
import time
import tempfile
import subprocess
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tools.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir

import psycopg2

ALLOWED_CPUS = list(range(20))
ALLOWED_CPU_MASK = ",".join(str(c) for c in ALLOWED_CPUS)
TIMING_LOG = "/tmp/pgdl_agent_timing.log"
DB_CONFIG = {
    "host": "localhost", "port": "5432", "user": "why",
    "dbname": "postgres", "password": "123456",
}
QUERY = (
    "SELECT count(*) FROM ("
    "SELECT unnest(db_agent_single('reasoning', sub_table.reasoning_vec)) AS score "
    "FROM (SELECT * FROM reasoning_vector_test ORDER BY id LIMIT {n}) AS sub_table) t;"
)

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

AGENT_LINE = re.compile(
    r"Agent:\s+(\S+)\s+Total:\s+([\d.]+)\s+ms,\s+Calls:\s+(\d+),\s+Average:\s+([\d.]+)\s+ms"
)
TOTAL_LINE = re.compile(r"Overall Total:\s+([\d.]+)\s+ms")


def gpu_snapshot():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True,
    )
    parts = [p.strip() for p in r.stdout.strip().split(",")]
    if len(parts) < 3:
        return {}
    return {
        "gpu": int(parts[0]),
        "mem_used_mib": float(parts[1]),
        "util_pct": float(parts[2]),
    }


class GpuLoad:
    def __init__(self):
        self.procs = []
        self.stop_file = None
        self.script_file = None

    def start(self, n_workers=2, mem_gb=2.0):
        self.stop_file = tempfile.mktemp(suffix=".stop")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as sf:
            sf.write(NEW_WORKER_SCRIPT)
            self.script_file = sf.name
        self.logs = []
        for i in range(n_workers):
            log = open(f"/tmp/breakdown_gpu_worker_{i}.log", "w")
            self.logs.append(log)
            p = subprocess.Popen(
                ["taskset", "-c", ALLOWED_CPU_MASK, sys.executable,
                 self.script_file, str(mem_gb), self.stop_file],
                stdout=log, stderr=subprocess.STDOUT,
            )
            self.procs.append(p)
        deadline = time.time() + 60
        snap = {}
        while time.time() < deadline:
            time.sleep(2)
            snap = gpu_snapshot()
            alive = sum(p.poll() is None for p in self.procs)
            print(f"  waiting for load: {snap} alive={alive}/{n_workers}", flush=True)
            if snap.get("mem_used_mib", 0) >= 2000 and alive == n_workers:
                return snap
        raise RuntimeError(f"GPU load did not come up: {snap}")

    def stop(self):
        if self.stop_file:
            try:
                with open(self.stop_file, "w") as f:
                    f.write("stop")
            except OSError:
                pass
        time.sleep(2)
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        for log in getattr(self, "logs", []):
            log.close()
        for path in (self.stop_file, self.script_file):
            if path and os.path.exists(path):
                os.remove(path)


def pin_backend(cur):
    cur.execute("SELECT pg_backend_pid()")
    pid = cur.fetchone()[0]
    subprocess.run(
        ["taskset", "-p", "-c", ALLOWED_CPU_MASK, str(pid)],
        capture_output=True, timeout=5,
    )
    return pid


def read_timing(since_mtime):
    if not os.path.exists(TIMING_LOG):
        return {"agents": {}, "overall_ms": None, "raw": ""}
    if os.path.getmtime(TIMING_LOG) + 1 < since_mtime:
        return {"agents": {}, "overall_ms": None, "raw": "", "stale": True}
    raw = open(TIMING_LOG).read()
    agents = {}
    overall = None
    for line in raw.splitlines():
        m = AGENT_LINE.search(line)
        if m:
            agents[m.group(1)] = {
                "total_ms": float(m.group(2)),
                "calls": int(m.group(3)),
                "avg_ms": float(m.group(4)),
            }
        m2 = TOTAL_LINE.search(line)
        if m2:
            overall = float(m2.group(1))
    return {"agents": agents, "overall_ms": overall, "raw": raw}


def run_query(n):
    started = time.time()
    wall0 = time.perf_counter()
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    backend_pid = pin_backend(cur)
    cur.execute("SELECT register_process();")
    cur.execute(QUERY.format(n=n))
    rows = cur.fetchall()
    conn.commit()
    conn.close()
    wall_ms = (time.perf_counter() - wall0) * 1000
    return {
        "wall_ms": round(wall_ms, 1),
        "rows": rows,
        "backend_pid": backend_pid,
        "timing": read_timing(started),
    }


def categorize(agents):
    dba = {
        "Inference": agents.get("EvaluationAgent", {}).get("total_ms", 0.0),
        "Execution": agents.get("ExecutionAgent", {}).get("total_ms", 0.0),
        "Orchestration": sum(
            agents.get(name, {}).get("total_ms", 0.0)
            for name in ("OrchestrationAgent", "OptimizationAgent",
                         "PerceptionAgent", "ScheduleAgent")
        ),
    }
    dba["Total"] = sum(dba.values())
    return {k: round(v, 3) for k, v in dba.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mem-gb", type=float, default=2.0)
    args = parser.parse_args()

    os.sched_setaffinity(0, ALLOWED_CPUS)
    print(f"low GPU load: {args.workers} x {args.mem_gb}GB on GPU 0, n={args.n}")
    print("gpu before", gpu_snapshot())

    load = GpuLoad()
    try:
        load.start(args.workers, args.mem_gb)
        print("gpu after workers", gpu_snapshot())
        print("--- warmup ---")
        warmup = run_query(args.n)
        print(f"warmup wall {warmup['wall_ms']:.1f} ms, agent total {warmup['timing'].get('overall_ms')}")
        time.sleep(2)
        print("--- timed ---")
        timed = run_query(args.n)
        print(f"timed wall {timed['wall_ms']:.1f} ms, agent total {timed['timing'].get('overall_ms')}")
        print(timed["timing"].get("raw", ""))
    finally:
        load.stop()

    paper = {
        "Inference": 52594, "Execution": 11695,
        "Orchestration": 1201, "Total": 65490,
    }
    scheduling_low_ms = 92047.6
    categories = categorize(timed["timing"]["agents"])
    result = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n": args.n,
            "load": "scheduling low GPU: 2 workers x 2GB, new-style, GPU 0",
            "cpus": "0-19",
            "warmup_runs": 1,
            "timed_runs": 1,
            "timing_enabled": True,
            "add_optimization_logic": False,
        },
        "warmup_wall_ms": warmup["wall_ms"],
        "timed_wall_ms": timed["wall_ms"],
        "agents": timed["timing"]["agents"],
        "overall_ms": timed["timing"]["overall_ms"],
        "categories_ms": categories,
        "paper_breakdown_ms": paper,
        "scheduling_low_wall_ms": scheduling_low_ms,
        "wall_vs_scheduling_ratio": round(timed["wall_ms"] / scheduling_low_ms, 3),
        "agent_total_vs_scheduling_ratio": (
            round(timed["timing"]["overall_ms"] / scheduling_low_ms, 3)
            if timed["timing"]["overall_ms"] else None
        ),
    }
    out_dir = ensure_dir(GENERAL_RESULTS_DIR)
    path = os.path.join(
        out_dir, f"breakdown_low_gpu_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print("saved", path)
    print("categories", categories)
    print(
        f"wall/scheduling={result['wall_vs_scheduling_ratio']} "
        f"agent_total/scheduling={result['agent_total_vs_scheduling_ratio']}"
    )


if __name__ == "__main__":
    main()
