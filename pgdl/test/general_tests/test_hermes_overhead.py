#!/usr/bin/env python3
"""

Quick Hermes overhead measurement - only wall time and memory.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


import os
import time
import json
import subprocess
import psutil
from datetime import datetime
from support.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "morphingdb_test", "models")
TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")

import numpy as np
import torch
import psycopg2

DB_CONFIG = {
    "dbname": "postgres",
    "host": "localhost",
    "port": "5432",
    "user": "why",
    "password": "123456"
}


def get_gpu_mem():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "2"],
        capture_output=True, text=True
    )
    return float(r.stdout.strip()) if r.returncode == 0 else 0


def measure_pipeline(name, load_data_func, run_func, warm=False):
    """Measure a pipeline's resource usage."""
    input_vecs, step2_vecs, step2_tensors = load_data_func()
    
    if warm:
        model_cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location="cpu")
        model_cross.eval()
        model_reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu")
        model_reader.eval()
        cross_batch = torch.from_numpy(np.stack([v[:2, :] for v in input_vecs], axis=0).astype(np.float32))
        with torch.no_grad():
            model_cross(cross_batch[:1])
            model_reader(step2_tensors[0])
        time.sleep(1)
    
    proc = psutil.Process(os.getpid())
    gpu_start = get_gpu_mem()
    cpu_start = proc.cpu_times()
    
    wall_start = time.perf_counter()
    run_func(input_vecs, step2_vecs, step2_tensors)
    wall_end = time.perf_counter()
    
    gpu_end = get_gpu_mem()
    cpu_end = proc.cpu_times()
    mem_mb = proc.memory_info().rss / (1024 * 1024)
    
    return {
        "wall_s": round(wall_end - wall_start, 2),
        "cpu_s": round((cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system), 2),
        "gpu_mem_mb": round(gpu_end - gpu_start, 0),
        "mem_mb": round(mem_mb, 0)
    }


def load_hotpotqa(n):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT get_mvec_data(reasoning_vec) FROM reasoning_vector_test LIMIT %s", (n,))
    input_vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    cur.execute("SELECT get_mvec_data(text_vec) FROM reasoning_vector_step2 ORDER BY id LIMIT %s", (2 * n,))
    step2_vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()
    
    step2_tensors = []
    for i in range(n):
        idx = i * 2
        t1 = np.stack([step2_vecs[idx][:2, :], step2_vecs[idx+1][:2, :]], axis=0).astype(np.float32)
        step2_tensors.append(torch.from_numpy(t1))
    
    return input_vecs, step2_vecs, step2_tensors


def run_hotpotqa(data):
    input_vecs, step2_vecs, step2_tensors = data
    cross_batch = torch.from_numpy(np.stack([v[:2, :] for v in input_vecs], axis=0).astype(np.float32))
    
    model_cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location="cpu")
    model_cross.eval()
    model_reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu")
    model_reader.eval()
    
    n = len(input_vecs)
    with torch.no_grad():
        scores = model_cross(cross_batch).squeeze(-1).tolist()
    
    for i in range(n):
        with torch.no_grad():
            model_reader(step2_tensors[i])
        
        res_str_parts = []
        for j in [0, 1]:
            vec_idx = i * 2 + j
            input_ids = step2_vecs[vec_idx][0, :].astype(int).tolist()
            res_str_parts.append(",".join(str(t) for t in input_ids[:4]))
        res_str = ";".join(res_str_parts)
        
        subprocess.run(
            ["python3", f"{TOOLS_DIR}/reader_decoder.py", "--ids", res_str],
            capture_output=True, text=True, cwd=TOOLS_DIR, timeout=30
        )


def load_musique(n):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT get_mvec_data(reasoning_vec) FROM musique_vector_test LIMIT %s", (n,))
    input_vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    cur.execute("SELECT get_mvec_data(text_vec) FROM musique_vector_step2 ORDER BY id")
    step2_vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()
    
    return input_vecs, step2_vecs, []


def run_musique(data):
    input_vecs, step2_vecs, _ = data
    cross_batch = torch.from_numpy(np.stack([v[:2, :] for v in input_vecs], axis=0).astype(np.float32))
    
    model_cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location="cpu")
    model_cross.eval()
    model_reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu")
    model_reader.eval()
    
    with torch.no_grad():
        scores = model_cross(cross_batch).squeeze(-1).tolist()
    
    top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:2]
    
    for orig_idx, _ in top_k:
        idx = orig_idx * 2
        for j in [idx, idx + 1]:
            vec = step2_vecs[j]
            reader_input = torch.from_numpy(vec[:2, :].astype(np.float32).reshape(1, 2, 128))
            with torch.no_grad():
                model_reader(reader_input)


def main():
    print("Hermes Overhead Measurement")
    print("=" * 60)
    
    results = {}
    
    # HotpotQA n=1
    print("\nHotpotQA n=1")
    data = load_hotpotqa(1)
    cold = measure_pipeline("hotpotqa_n1", lambda: load_hotpotqa(1), run_hotpotqa, warm=False)
    print(f"  Cold: Wall={cold['wall_s']}s CPU={cold['cpu_s']}s GPU={cold['gpu_mem_mb']}MB MEM={cold['mem_mb']}MB")
    cached = measure_pipeline("hotpotqa_n1", lambda: load_hotpotqa(1), run_hotpotqa, warm=True)
    print(f"  Cached: Wall={cached['wall_s']}s CPU={cached['cpu_s']}s GPU={cached['gpu_mem_mb']}MB MEM={cached['mem_mb']}MB")
    results["hotpotqa_n1"] = {"cold": cold, "cached": cached}
    
    # HotpotQA n=20
    print("\nHotpotQA n=20")
    data = load_hotpotqa(20)
    cold = measure_pipeline("hotpotqa_n20", lambda: load_hotpotqa(20), run_hotpotqa, warm=False)
    print(f"  Cold: Wall={cold['wall_s']}s CPU={cold['cpu_s']}s GPU={cold['gpu_mem_mb']}MB MEM={cold['mem_mb']}MB")
    cached = measure_pipeline("hotpotqa_n20", lambda: load_hotpotqa(20), run_hothotqa, warm=True)
    print(f"  Cached: Wall={cached['wall_s']}s CPU={cached['cpu_s']}s GPU={cached['gpu_mem_mb']}MB MEM={cached['mem_mb']}MB")
    results["hotpotqa_n20"] = {"cold": cold, "cached": cached}
    
    # musique n=1
    print("\nmusique n=1")
    data = load_musique(1)
    cold = measure_pipeline("musique_n1", lambda: load_musique(1), run_musique, warm=False)
    print(f"  Cold: Wall={cold['wall_s']}s CPU={cold['cpu_s']}s GPU={cold['gpu_mem_mb']}MB MEM={cold['mem_mb']}MB")
    cached = measure_pipeline("musique_n1", lambda: load_musique(1), run_musique, warm=True)
    print(f"  Cached: Wall={cached['wall_s']}s CPU={cached['cpu_s']}s GPU={cached['gpu_mem_mb']}MB MEM={cached['mem_mb']}MB")
    results["musique_n1"] = {"cold": cold, "cached": cached}
    
    # musique n=20
    print("\nmusique n=20")
    data = load_musique(20)
    cold = measure_pipeline("musique_n20", lambda: load_musique(20), run_musique, warm=False)
    print(f"  Cold: Wall={cold['wall_s']}s CPU={cold['cpu_s']}s GPU={cold['gpu_mem_mb']}MB MEM={cold['mem_mb']}MB")
    cached = measure_pipeline("musique_n20", lambda: load_musique(20), run_musique, warm=True)
    print(f"  Cached: Wall={cached['wall_s']}s CPU={cached['cpu_s']}s GPU={cached['gpu_mem_mb']}MB MEM={cached['mem_mb']}MB")
    results["musique_n20"] = {"cold": cold, "cached": cached}
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = ensure_dir(GENERAL_RESULTS_DIR)
    with open(os.path.join(results_dir, f"hermes_overhead_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to results/hermes_overhead_{ts}.json")


if __name__ == "__main__":
    main()
