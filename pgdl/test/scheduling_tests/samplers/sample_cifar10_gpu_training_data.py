#!/usr/bin/env python3
"""
Sample cifar10 query timing on GPU under different GPU load levels for training the GPU load predictor.

cifar10 (googlenet, input 1x3x224x224) is a GPU-favorable task:
  - GPU low load: GPU is fast
  - GPU mid/high load: CPU becomes competitive

GPU load levels use background CUDA matmul workers on GPU 0:
  Low:    2 workers, each 2GB tensor  (~4GB GPU memory, ~40-60% utilization)
  Medium: 4 workers, each 2GB tensor  (~8GB GPU memory, ~70-85% utilization)
  High:   8 workers, each 1.5GB tensor (~12GB GPU memory, ~90%+ utilization)

Data shape: (1, 3, 224, 224) = cifar10 image vector
Model: googlenet_cifar10 (13M params, 52MB)
"""
import os
import sys
import time
import subprocess
import multiprocessing
import psycopg2

ALLOWED_CPUS = list(range(20))
ALLOWED_CPU_MASK = ",".join(str(c) for c in ALLOWED_CPUS)
DB_CONFIG = {
    "host": "localhost", "port": "5432", "user": "why",
    "dbname": "postgres", "password": "123456",
}

# GPU load levels: (n_workers, mem_gb_per_worker)
# Keep total well below 22GB free on GPU 0 (leave 4GB headroom for inference + safety)
GPU_LOAD_LEVELS = [
    ("low",    2, 2.0),   # 4GB total
    ("medium", 4, 2.0),   # 8GB total
    ("high",   8, 1.5),   # 12GB total
]

ROWS_PER_QUERY = 128   # 4 windows of WINDOW_SIZE=32
QUERIES_PER_LEVEL = 10

# googlenet_cifar10: input shape (1,3,224,224)
# MAC count: only .weight params (not .bias), matches C++ AnalyzeModel
# conv2d: out*in*kH*kW*224*224; linear: out*in; bn.weight: numel
MODEL_MAC_COUNT = 286936923504
MODEL_PARAM_COUNT = 13004888
MODEL_PARAM_SIZE = 52019552
DATA_SHAPE_1 = 1
DATA_SHAPE_2 = 3
DATA_SHAPE_3 = 224
DATA_SHAPE_4 = 224

CIFAR_QUERY = (
    "SELECT predict_batch_float8('googlenet_cifar10', 'gpu:0', image_vector) "
    "OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) "
    "FROM cifar_image_vector_table LIMIT {row_count};"
)

# Background GPU worker: allocate tensor on GPU 0 and run continuous matmul
GPU_WORKER_SCRIPT = """
import torch
import time
import sys
import os

os.sched_setaffinity(0, list(range(20)))
mem_gb = float(sys.argv[1])
stop_file = sys.argv[2]

# Allocate on GPU 0
n = int((mem_gb * 1024**3 / 4) ** 0.5)  # sqrt of (bytes / 4 bytes per float32)
n = min(n, 8192)  # cap at 8192 to keep matmul manageable
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
    subprocess.run(
        ["taskset", "-p", "-c", ALLOWED_CPU_MASK, str(pid)],
        capture_output=True, timeout=5
    )


def get_gpu_metrics():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=0",
             "--query-gpu=clocks.max.sm,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        parts = [p.strip() for p in result.stdout.strip().split(',')]
        gpu_freq = float(parts[0])
        util = float(parts[1])
        mem_used = float(parts[2])
        mem_total = float(parts[3])
        # RTX 3090 = 8.6 -> 10496 CUDA cores
        cuda_cores = 10496.0
        return cuda_cores, gpu_freq, util, mem_used, mem_total
    except Exception as e:
        print(f"  Warning: nvidia-smi failed: {e}")
        return 10496.0, 1695.0, 0.0, 0.0, 24576.0


def run_query_and_collect(conn, row_count):
    cur = conn.cursor()
    get_backend_pid_and_pin(cur)
    cur.execute("SELECT register_process()")
    cur.execute("SET statement_timeout = '120s'")

    cuda_cores, gpu_freq, util, mem_used, mem_total = get_gpu_metrics()

    query = CIFAR_QUERY.format(row_count=row_count)
    start = time.perf_counter()
    try:
        cur.execute(query)
        cur.fetchall()
    except Exception as e:
        conn.rollback()
        return {'status': 'error', 'error': str(e)}

    elapsed_us = int((time.perf_counter() - start) * 1e6)

    n_windows = row_count // 32
    per_window_us = elapsed_us // max(n_windows, 1)

    for _ in range(n_windows):
        cur.execute(
            "INSERT INTO cifar10_gpu_load_training_data "
            "(cuda_cores, gpu_freq, util, mem_used, mem_total, "
            " execution_runtime_us, "
            " data_shape_1, data_shape_2, data_shape_3, data_shape_4, "
            " model_mac_count, model_param_count, model_param_size) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (cuda_cores, gpu_freq, util, mem_used, mem_total,
             per_window_us,
             DATA_SHAPE_1, DATA_SHAPE_2, DATA_SHAPE_3, DATA_SHAPE_4,
             MODEL_MAC_COUNT, MODEL_PARAM_COUNT, MODEL_PARAM_SIZE)
        )

    conn.commit()
    return {
        'status': 'success',
        'latency_ms': round(elapsed_us / 1000, 1),
        'windows': n_windows,
        'util': util,
        'mem_used': mem_used,
    }


def get_training_count():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM cifar10_gpu_load_training_data")
    count = cur.fetchone()[0]
    conn.close()
    return count


def start_gpu_workers(n_workers, mem_gb_per_worker):
    """Start background GPU compute workers via subprocess."""
    import tempfile
    stop_file = tempfile.mktemp(suffix='.stop')
    worker_script = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    worker_script.write(GPU_WORKER_SCRIPT)
    worker_script.flush()
    worker_script.close()

    procs = []
    for _ in range(n_workers):
        p = subprocess.Popen(
            ["taskset", "-c", ALLOWED_CPU_MASK,
             sys.executable, worker_script.name,
             str(mem_gb_per_worker), stop_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        procs.append(p)

    return procs, stop_file, worker_script.name


def stop_gpu_workers(procs, stop_file, script_file):
    """Signal workers to stop and clean up."""
    # Create stop file to signal workers
    with open(stop_file, 'w') as f:
        f.write('stop')
    time.sleep(2)
    for p in procs:
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    try:
        os.unlink(stop_file)
    except Exception:
        pass
    try:
        os.unlink(script_file)
    except Exception:
        pass


def main():
    os.sched_setaffinity(0, ALLOWED_CPUS)
    print("=" * 60)
    print("  cifar10 GPU Training Data Collection")
    print("=" * 60)
    print(f"  GPU load levels: {[(n, w, m) for n, w, m in GPU_LOAD_LEVELS]}")
    print(f"  Queries per level: {QUERIES_PER_LEVEL}")
    print(f"  Rows per query: {ROWS_PER_QUERY}")
    print()

    initial_count = get_training_count()
    print(f"  Initial training data count: {initial_count}")
    print()

    # --- Phase 0: No GPU load (baseline, low GPU util) ---
    print(f"\n{'='*60}")
    print(f"  GPU LOAD LEVEL: BASELINE (no workers, low GPU util)")
    print(f"{'='*60}")
    _, _, util_base, mem_base, _ = get_gpu_metrics()
    print(f"  GPU util: {util_base:.0f}%  mem_used: {mem_base:.0f}MB")
    conn = get_conn()
    for i in range(QUERIES_PER_LEVEL):
        r = run_query_and_collect(conn, ROWS_PER_QUERY)
        if r['status'] == 'success':
            print(f"    Query {i+1:2d}: {r['latency_ms']:.1f}ms | "
                  f"util={r['util']:.0f}% mem={r['mem_used']:.0f}MB | "
                  f"{r['windows']} windows inserted")
        else:
            print(f"    Query {i+1:2d}: ERROR - {r.get('error', 'unknown')}")
        time.sleep(0.5)
    conn.close()
    print("  Cooling down for 5s...")
    time.sleep(5)

    for level_name, n_workers, mem_gb in GPU_LOAD_LEVELS:
        print(f"\n{'='*60}")
        print(f"  GPU LOAD LEVEL: {level_name.upper()}")
        print(f"  ({n_workers} workers x {mem_gb}GB = {n_workers*mem_gb:.0f}GB on GPU 0)")
        print(f"{'='*60}")

        procs, stop_file, script_file = start_gpu_workers(n_workers, mem_gb)
        print(f"  Started {n_workers} GPU workers, waiting 5s to stabilize...")
        time.sleep(5)

        _, _, util, mem_used, _ = get_gpu_metrics()
        print(f"  GPU util: {util:.0f}%  mem_used: {mem_used:.0f}MB")

        conn = get_conn()
        for i in range(QUERIES_PER_LEVEL):
            r = run_query_and_collect(conn, ROWS_PER_QUERY)
            if r['status'] == 'success':
                print(f"    Query {i+1:2d}: {r['latency_ms']:.1f}ms | "
                      f"util={r['util']:.0f}% mem={r['mem_used']:.0f}MB | "
                      f"{r['windows']} windows inserted")
            else:
                print(f"    Query {i+1:2d}: ERROR - {r.get('error', 'unknown')}")
            time.sleep(0.5)
        conn.close()

        stop_gpu_workers(procs, stop_file, script_file)
        print("  Cooling down for 10s...")
        time.sleep(10)
        _, _, util_after, _, _ = get_gpu_metrics()
        print(f"  GPU util after cooldown: {util_after:.0f}%")

        current_count = get_training_count()
        print(f"  Training data count: {current_count}")

    final_count = get_training_count()
    new_count = final_count - initial_count
    print(f"\n{'='*60}")
    print(f"  COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  New rows: {new_count}, Total rows: {final_count}")

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            round(util::numeric, 0) as util_bucket,
            count(*) as rows,
            round(avg(execution_runtime_us)::numeric, 0) as avg_us,
            round(min(execution_runtime_us)::numeric, 0) as min_us,
            round(max(execution_runtime_us)::numeric, 0) as max_us
        FROM cifar10_gpu_load_training_data
        GROUP BY 1 ORDER BY 1
    """)
    rows = cur.fetchall()
    print(f"\n  Data by GPU utilization bucket:")
    print(f"  {'Util%':>6s} | {'Rows':>5s} | {'Avg_us':>10s} | {'Min_us':>8s} | {'Max_us':>8s}")
    for row in rows:
        print(f"  {row[0]:>6.0f} | {row[1]:5d} | {row[2]:>10.0f} | {row[3]:>8.0f} | {row[4]:>8.0f}")
    conn.close()


if __name__ == "__main__":
    main()
