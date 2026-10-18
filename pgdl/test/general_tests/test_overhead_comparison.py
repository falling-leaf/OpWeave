#!/usr/bin/env python3
"""

Simple overhead measurement for DBAgent and Hermes on reasoning tasks.
Uses time module for wall-clock, resource module for CPU time,
and nvidia-smi for GPU memory tracking.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


import json
import time
import subprocess
import resource
import psycopg2
from datetime import datetime

# Configuration
DB_CONFIG = {
    "dbname": "postgres",
    "host": "localhost",
    "port": "5432",
    "user": "why",
    "password": "123456"
}
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "morphingdb_test", "models")
TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
from tools.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir

RESULTS_DIR = ensure_dir(GENERAL_RESULTS_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def get_gpu_memory():
    """Get GPU memory usage in MB."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits", "-i", "2"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return float(result.stdout.strip())
    return 0


def get_cpu_time():
    """Get CPU time in seconds (user + system)."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def run_dbagent_query(dataset, n, warm=False):
    """Run DBAgent query and return measurements."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    if dataset == "hotpotqa":
        table = "reasoning_vector_test"
        task = "reasoning"
    else:
        table = "musique_vector_test"
        task = "musique"
    
    # Warm up if needed
    if warm:
        cur.execute(
            f"SELECT unnest(db_agent_single('{task}', sub_table.reasoning_vec)) AS score "
            f"FROM (SELECT * FROM {table} LIMIT 1) AS sub_table;"
        )
        cur.fetchall()
        time.sleep(1)  # Let things settle
    
    # Get initial measurements
    gpu_mem_start = get_gpu_memory()
    cpu_start = get_cpu_time()
    
    # Run query
    sql = (
        f"SELECT unnest(db_agent_single('{task}', sub_table.reasoning_vec)) AS score "
        f"FROM (SELECT * FROM {table} LIMIT {n}) AS sub_table;"
    )
    
    wall_start = time.perf_counter()
    cur.execute(sql)
    rows = cur.fetchall()
    wall_end = time.perf_counter()
    
    # Get final measurements
    gpu_mem_end = get_gpu_memory()
    cpu_end = get_cpu_time()
    
    conn.close()
    
    wall_s = wall_end - wall_start
    cpu_s = cpu_end - cpu_start
    gpu_mem_used = gpu_mem_end - gpu_mem_start
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB
    
    return {
        "wall_s": round(wall_s, 2),
        "cpu_s": round(cpu_s, 2),
        "gpu_mem_mb": round(gpu_mem_used, 0),
        "mem_mb": round(mem_mb, 0),
        "rows": len(rows)
    }


def run_hermes_pipeline(dataset, n, warm=False):
    """Run Hermes pipeline and return measurements."""
    import numpy as np
    import torch
    
    # Load data
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    if dataset == "hotpotqa":
        cur.execute("SELECT get_mvec_data(reasoning_vec) FROM reasoning_vector_test LIMIT %s", (n,))
        cur.execute("SELECT get_mvec_data(text_vec) FROM reasoning_vector_step2 ORDER BY id LIMIT %s", (2 * n,))
    else:
        cur.execute("SELECT get_mvec_data(reasoning_vec) FROM musique_vector_test LIMIT %s", (n,))
        cur.execute("SELECT get_mvec_data(text_vec) FROM musique_vector_step2 ORDER BY id")
    
    input_vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    step2_vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()
    
    cross_batch = torch.from_numpy(np.stack([v[:2, :] for v in input_vecs], axis=0).astype(np.float32))
    step2_tensors = []
    for i in range(n):
        idx = i * 2
        t1 = np.stack([step2_vecs[idx][:2, :], step2_vecs[idx+1][:2, :]], axis=0).astype(np.float32)
        step2_tensors.append(torch.from_numpy(t1))
    
    # Warm up if needed
    if warm:
        model_cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location="cpu")
        model_cross.eval()
        with torch.no_grad():
            model_cross(cross_batch[:1])
        time.sleep(1)
    
    # Get initial measurements
    gpu_mem_start = get_gpu_memory()
    cpu_start = get_cpu_time()
    
    # Run pipeline
    wall_start = time.perf_counter()
    
    model_cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location="cpu")
    model_cross.eval()
    model_reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu")
    model_reader.eval()
    
    with torch.no_grad():
        scores = model_cross(cross_batch).squeeze(-1).tolist()
    
    # Deberta reader + decoder
    for i in range(n):
        with torch.no_grad():
            reader_output = model_reader(step2_tensors[i])
        start_logits, end_logits = reader_output[:, 0, :], reader_output[:, 1, :]
        
        res_str_parts = []
        for j in [0, 1]:
            vec_idx = i * 2 + j
            input_ids = step2_vecs[vec_idx][0, :].astype(int).tolist()
            s = int(start_logits[j].argmax().item())
            e = int(end_logits[j].argmax().item())
            res_str_parts.append(",".join(str(t) for t in input_ids[s:e+1]))
        res_str = ";".join(res_str_parts)
        
        subprocess.run(
            ["python3", f"{TOOLS_DIR}/reader_decoder.py", "--ids", res_str],
            capture_output=True, text=True, cwd=TOOLS_DIR, timeout=60
        )
    
    wall_end = time.perf_counter()
    
    # Get final measurements
    gpu_mem_end = get_gpu_memory()
    cpu_end = get_cpu_time()
    
    wall_s = wall_end - wall_start
    cpu_s = cpu_end - cpu_start
    gpu_mem_used = gpu_mem_end - gpu_mem_start
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    
    return {
        "wall_s": round(wall_s, 2),
        "cpu_s": round(cpu_s, 2),
        "gpu_mem_mb": round(gpu_mem_used, 0),
        "mem_mb": round(mem_mb, 0)
    }


def main():
    print("=" * 80)
    print("Overhead Comparison: DBAgent vs Hermes (Reasoning Tasks)")
    print("=" * 80)
    print()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "overhead_comparison",
        "datasets": {}
    }
    
    configs = [
        ("hotpotqa", [1, 20]),
        ("musique", [1, 20]),
    ]
    
    for dataset, n_values in configs:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        
        results["datasets"][dataset] = {}
        
        for n in n_values:
            print(f"\n--- n={n} ---")
            
            # DBAgent
            print("\n[DBAgent] Cold start...")
            dbagent_cold = run_dbagent_query(dataset, n, warm=False)
            print(f"  Wall: {dbagent_cold['wall_s']}s, CPU: {dbagent_cold['cpu_s']}s, "
                  f"GPU Mem: {dbagent_cold['gpu_mem_mb']}MB, MEM: {dbagent_cold['mem_mb']}MB")
            
            print("[DBAgent] Cached (warm)...")
            dbagent_cached = run_dbagent_query(dataset, n, warm=True)
            print(f"  Wall: {dbagent_cached['wall_s']}s, CPU: {dbagent_cached['cpu_s']}s, "
                  f"GPU Mem: {dbagent_cached['gpu_mem_mb']}MB, MEM: {dbagent_cached['mem_mb']}MB")
            
            # Hermes
            print("\n[Hermes] Cold start...")
            hermes_cold = run_hermes_pipeline(dataset, n, warm=False)
            print(f"  Wall: {hermes_cold['wall_s']}s, CPU: {hermes_cold['cpu_s']}s, "
                  f"GPU Mem: {hermes_cold['gpu_mem_mb']}MB, MEM: {hermes_cold['mem_mb']}MB")
            
            print("[Hermes] Cached (warm)...")
            hermes_cached = run_hermes_pipeline(dataset, n, warm=True)
            print(f"  Wall: {hermes_cached['wall_s']}s, CPU: {hermes_cached['cpu_s']}s, "
                  f"GPU Mem: {hermes_cached['gpu_mem_mb']}MB, MEM: {hermes_cached['mem_mb']}MB")
            
            results["datasets"][dataset][f"n{n}"] = {
                "dbagent_cold": dbagent_cold,
                "dbagent_cached": dbagent_cached,
                "hermes_cold": hermes_cold,
                "hermes_cached": hermes_cached
            }
    
    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(RESULTS_DIR, f"overhead_comparison_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to {json_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
