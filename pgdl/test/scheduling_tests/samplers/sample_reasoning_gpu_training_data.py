#!/usr/bin/env python3
"""
Sample reasoning (cross_encoder) query timing on GPU under different GPU load
levels for training the GPU load predictor.

GPU load levels use background CUDA matmul workers on GPU 0 (same canonical
config as the cifar10/DBAgent/MorphingDB scheduling tests):
  Low:    2 workers x 2GB   new-style (no sync, 50ms sleep)
  Medium: 3 workers x 2GB   new-style
  High:   8 workers x 1.5GB oom-style (sync + per-iter alloc, 1ms sleep)

Data shape: (1, 4, 128)  -- reasoning_vec mvec shape
Model: cross_encoder (22.7M params, ~91MB)
"""
import os
import sys
import time
import subprocess
import psycopg2

ALLOWED_CPUS = list(range(20))
ALLOWED_CPU_MASK = ",".join(str(c) for c in ALLOWED_CPUS)
DB_CONFIG = {
    "host": "localhost", "port": "5432", "user": "why",
    "dbname": "postgres", "password": "123456",
}

# (name, n_workers, mem_gb, oom_style)
GPU_LOAD_LEVELS = [
    ("low",    2, 2.0, False),
    ("medium", 3, 2.0, False),
    ("high",   8, 1.5, True),
]

ROW_COUNT = 20
WINDOW = 16
QUERIES_PER_LEVEL = 12

MODEL_MAC_COUNT = 22682496
MODEL_PARAM_COUNT = 22713601
MODEL_PARAM_SIZE = 90854404
DATA_SHAPE_1 = 1
DATA_SHAPE_2 = 4
DATA_SHAPE_3 = 128
DATA_SHAPE_4 = 0

REASONING_QUERY = (
    "SELECT predict_batch_float8('cross_encoder', 'gpu:0', reasoning_vec) "
    "OVER (ROWS BETWEEN CURRENT ROW AND {win} FOLLOWING) "
    "FROM reasoning_vector_test LIMIT {row_count};"
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


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def get_backend_pid_and_pin(cur):
    cur.execute("SELECT pg_backend_pid()")
    pid = cur.fetchone()[0]
    subprocess.run(["taskset", "-p", "-c", ALLOWED_CPU_MASK, str(pid)],
                   capture_output=True, timeout=5)


def get_gpu_metrics():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=0",
             "--query-gpu=clocks.max.sm,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        parts = [p.strip() for p in result.stdout.strip().split(',')]
        gpu_freq = float(parts[0]); util = float(parts[1])
        mem_used = float(parts[2]); mem_total = float(parts[3])
        cuda_cores = 10496.0  # RTX 3090 (8.6)
        return cuda_cores, gpu_freq, util, mem_used, mem_total
    except Exception as e:
        print(f"  Warning: nvidia-smi failed: {e}")
        return 10496.0, 1695.0, 0.0, 0.0, 24576.0


def run_query_and_collect(conn):
    cur = conn.cursor()
    get_backend_pid_and_pin(cur)
    cur.execute("SELECT register_process()")
    cur.execute("SET statement_timeout = '120s'")

    cuda_cores, gpu_freq, util, mem_used, mem_total = get_gpu_metrics()

    query = REASONING_QUERY.format(win=WINDOW - 1, row_count=ROW_COUNT)
    start = time.perf_counter()
    try:
        cur.execute(query)
        cur.fetchall()
    except Exception as e:
        conn.rollback()
        return {'status': 'error', 'error': str(e)}

    elapsed_us = int((time.perf_counter() - start) * 1e6)

    cur.execute(
        "INSERT INTO reasoning_gpu_load_training_data "
        "(cuda_cores, gpu_freq, util, mem_used, mem_total, execution_runtime_us, "
        " data_shape_1, data_shape_2, data_shape_3, data_shape_4, "
        " model_mac_count, model_param_count, model_param_size) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (cuda_cores, gpu_freq, util, mem_used, mem_total, elapsed_us,
         DATA_SHAPE_1, DATA_SHAPE_2, DATA_SHAPE_3, DATA_SHAPE_4,
         MODEL_MAC_COUNT, MODEL_PARAM_COUNT, MODEL_PARAM_SIZE)
    )
    conn.commit()
    return {'status': 'success', 'latency_ms': round(elapsed_us / 1000, 1),
            'util': util, 'mem_used': mem_used}


def get_training_count():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM reasoning_gpu_load_training_data")
    count = cur.fetchone()[0]
    conn.close()
    return count


def start_gpu_workers(n_workers, mem_gb, oom_style):
    import tempfile
    stop_file = tempfile.mktemp(suffix='.stop')
    sf = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    sf.write(OOM_WORKER_SCRIPT if oom_style else NEW_WORKER_SCRIPT)
    sf.flush(); sf.close()
    procs = []
    for _ in range(n_workers):
        p = subprocess.Popen(
            ["taskset", "-c", ALLOWED_CPU_MASK, sys.executable, sf.name,
             str(mem_gb), stop_file],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)
    return procs, stop_file, sf.name


def stop_gpu_workers(procs, stop_file, script_file):
    with open(stop_file, 'w') as f:
        f.write('stop')
    time.sleep(2)
    for p in procs:
        try:
            p.terminate(); p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    for f in [stop_file, script_file]:
        try:
            os.unlink(f)
        except Exception:
            pass


def collect_level(label, n_workers=0, mem_gb=0.0, oom_style=False):
    print(f"\n{'='*60}")
    print(f"  GPU LOAD LEVEL: {label.upper()}")
    if n_workers:
        style = "oom-style" if oom_style else "new-style"
        print(f"  ({n_workers} workers x {mem_gb}GB = {n_workers*mem_gb:.0f}GB, {style})")
    print(f"{'='*60}")

    procs = stop_file = script_file = None
    if n_workers:
        procs, stop_file, script_file = start_gpu_workers(n_workers, mem_gb, oom_style)
        print(f"  Started {n_workers} GPU workers, stabilizing 8s...")
        time.sleep(8)

    _, _, util, mem_used, _ = get_gpu_metrics()
    print(f"  GPU util: {util:.0f}%  mem_used: {mem_used:.0f}MB")

    conn = get_conn()
    for i in range(QUERIES_PER_LEVEL):
        r = run_query_and_collect(conn)
        if r['status'] == 'success':
            print(f"    Query {i+1:2d}: {r['latency_ms']:.1f}ms | util={r['util']:.0f}% mem={r['mem_used']:.0f}MB")
        else:
            print(f"    Query {i+1:2d}: ERROR - {r.get('error', 'unknown')}")
        time.sleep(0.3)
    conn.close()

    if n_workers:
        stop_gpu_workers(procs, stop_file, script_file)
        print("  Cooling down 10s...")
        time.sleep(10)


def main():
    os.sched_setaffinity(0, ALLOWED_CPUS)
    print("=" * 60)
    print("  reasoning (cross_encoder) GPU Training Data Collection")
    print("=" * 60)

    initial_count = get_training_count()
    print(f"  Initial count: {initial_count}")

    # Baseline (no load) first
    collect_level("baseline")
    time.sleep(5)

    for name, nw, mem, oom in GPU_LOAD_LEVELS:
        collect_level(name, nw, mem, oom)

    final_count = get_training_count()
    print(f"\n{'='*60}")
    print(f"  COLLECTION COMPLETE: {final_count - initial_count} new, {final_count} total")
    print(f"{'='*60}")

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT round(util::numeric, 0) as util_bucket, count(*) as rows,
               round(avg(execution_runtime_us)::numeric, 0) as avg_us,
               round(min(execution_runtime_us)::numeric, 0) as min_us,
               round(max(execution_runtime_us)::numeric, 0) as max_us
        FROM reasoning_gpu_load_training_data GROUP BY 1 ORDER BY 1
    """)
    print(f"\n  {'Util%':>6s} | {'Rows':>5s} | {'Avg_us':>10s} | {'Min_us':>8s} | {'Max_us':>8s}")
    for row in cur.fetchall():
        print(f"  {row[0]:>6.0f} | {row[1]:5d} | {row[2]:>10.0f} | {row[3]:>8.0f} | {row[4]:>8.0f}")
    conn.close()


if __name__ == "__main__":
    main()
