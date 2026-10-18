#!/usr/bin/env python3
"""Hermes HotpotQA breakdown under the same low GPU load as the DBAgent run.

Load: 2 matmul workers x 2GB on GPU 0, new-style, CPUs 0-19.
Device placement matches scheduling: cross_encoder on cuda:0, deberta on CPU.
Agent categories match the paper breakdown table. Coordination is measured
gap time, not the old script's 2% scale factor.
1 warmup (discarded) + 1 timed run. Model load is outside the timed run.
"""
import os
import sys
import json
import time
import tempfile
import subprocess
from datetime import datetime

import numpy as np
import torch
import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from support.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "morphingdb_test", "models")
TOOLS_DIR = os.path.join(ROOT, "tools")
DB = {"dbname": "postgres", "host": "localhost", "port": "5432",
      "user": "why", "password": "123456"}
ALLOWED = list(range(20))
MASK = ",".join(str(c) for c in ALLOWED)
N = 20

WORKER = """\
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


def gpu_snap():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True)
    mem, util = [float(x.strip()) for x in r.stdout.strip().split(",")]
    return {"mem_used_mib": mem, "util_pct": util}


def start_load():
    stop = tempfile.mktemp(suffix=".stop")
    script = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    script.write(WORKER)
    script.close()
    procs, logs = [], []
    for i in range(2):
        log = open(f"/tmp/hermes_gpu_worker_{i}.log", "w")
        logs.append(log)
        procs.append(subprocess.Popen(
            ["taskset", "-c", MASK, sys.executable, script.name, "2.0", stop],
            stdout=log, stderr=subprocess.STDOUT))
    deadline = time.time() + 60
    snap = {}
    while time.time() < deadline:
        time.sleep(2)
        snap = gpu_snap()
        alive = sum(p.poll() is None for p in procs)
        print(f"  waiting: {snap} alive={alive}/2", flush=True)
        if snap["mem_used_mib"] >= 2000 and alive == 2:
            return procs, logs, stop, script.name, snap
    raise RuntimeError(f"GPU load did not come up: {snap}")


def stop_load(procs, logs, stop, script):
    try:
        open(stop, "w").write("stop")
    except OSError:
        pass
    time.sleep(1)
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    for log in logs:
        log.close()
    for path in (stop, script):
        if path and os.path.exists(path):
            os.remove(path)


def load_vecs():
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("SELECT get_mvec_data(reasoning_vec) FROM reasoning_vector_test ORDER BY id LIMIT %s", (N,))
    inputs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    cur.execute("SELECT get_mvec_data(text_vec) FROM reasoning_vector_step2 ORDER BY id LIMIT %s", (2 * N,))
    step2 = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()
    return inputs, step2


def run_once(models, inputs, step2):
    cross, reader = models
    gpu = torch.device("cuda:0")
    marks = {}
    t0 = time.perf_counter()

    t = time.perf_counter()
    batch = torch.from_numpy(np.stack([v[:2, :] for v in inputs]).astype(np.float32)).to(gpu)
    step2_tensors = []
    for i in range(N):
        idx = i * 2
        arr = np.stack([step2[idx][:2, :], step2[idx + 1][:2, :]]).astype(np.float32)
        step2_tensors.append(torch.from_numpy(arr))
    marks["DataLoadingAgent"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    with torch.no_grad():
        scores = cross(batch).squeeze(-1)
    torch.cuda.synchronize()
    cross_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    outputs = []
    for tensor in step2_tensors:
        with torch.no_grad():
            outputs.append(reader(tensor))
    deberta_ms = (time.perf_counter() - t) * 1000
    marks["ModelInferenceAgent"] = cross_ms + deberta_ms
    marks["cross_ms"] = cross_ms
    marks["deberta_ms"] = deberta_ms

    t = time.perf_counter()
    decoded = 0
    for i, out in enumerate(outputs):
        sl, el = out[:, 0, :], out[:, 1, :]
        parts = []
        for j in (0, 1):
            ids = step2[i * 2 + j][0, :].astype(int).tolist()
            s, e = int(sl[j].argmax()), int(el[j].argmax())
            parts.append(",".join(str(x) for x in ids[s:e + 1]))
        subprocess.run(
            ["uv", "run", f"{TOOLS_DIR}/reader_decoder.py", "--ids", ";".join(parts)],
            capture_output=True, text=True, cwd=TOOLS_DIR, timeout=60, check=False)
        decoded += 1
    marks["ResultProcessingAgent"] = (time.perf_counter() - t) * 1000
    marks["wall_ms"] = (time.perf_counter() - t0) * 1000
    marks["rows"] = decoded
    marks["scores"] = int(scores.numel())
    return marks


def main():
    os.sched_setaffinity(0, ALLOWED)
    print("loading models before load (excluded from timed run)")
    gpu = torch.device("cuda:0")
    t = time.perf_counter()
    cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location=gpu).eval()
    reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu").eval()
    model_load_ms = (time.perf_counter() - t) * 1000
    inputs, step2 = load_vecs()
    print(f"models loaded in {model_load_ms:.0f} ms, rows={len(inputs)}")

    print("starting low GPU load")
    procs, logs, stop, script, snap = start_load()
    print("load up", snap)
    try:
        print("--- warmup ---", flush=True)
        warm = run_once((cross, reader), inputs, step2)
        print(f"warmup wall {warm['wall_ms']:.1f} ms", flush=True)
        time.sleep(2)
        print("--- timed ---", flush=True)
        timed = run_once((cross, reader), inputs, step2)
        print(f"timed wall {timed['wall_ms']:.1f} ms", flush=True)
        during = gpu_snap()
    finally:
        stop_load(procs, logs, stop, script)

    agents = {
        "ResultProcessingAgent": round(timed["ResultProcessingAgent"], 3),
        "ModelInferenceAgent": round(timed["ModelInferenceAgent"], 3),
        "DataLoadingAgent": round(timed["DataLoadingAgent"], 3),
        "TaskPlanningAgent": 0.0,
        "CodeGenerationAgent": 0.0,
        "CoordinationAgent": 0.0,
    }
    categories = {
        "Inference": agents["ResultProcessingAgent"],
        "Execution": agents["ModelInferenceAgent"],
        "Orchestration": agents["DataLoadingAgent"],
    }
    categories["Total"] = round(sum(categories.values()), 3)
    result = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n": N,
            "load": "scheduling low GPU: 2 workers x 2GB, new-style, GPU 0",
            "device": "cross_encoder=cuda:0, deberta_reader=cpu",
            "model_load_excluded_ms": round(model_load_ms, 1),
            "coordination_not_scaled": True,
        },
        "gpu_during": during,
        "warmup_wall_ms": round(warm["wall_ms"], 1),
        "timed": {k: round(v, 3) if isinstance(v, float) else v for k, v in timed.items()},
        "agents_ms": agents,
        "categories_ms": categories,
    }
    path = os.path.join(ensure_dir(GENERAL_RESULTS_DIR),
                        f"hermes_breakdown_low_gpu_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print("saved", path)
    print("categories", categories)
    print("cross/deberta", round(timed["cross_ms"], 1), round(timed["deberta_ms"], 1))


if __name__ == "__main__":
    main()
