#!/usr/bin/env python3
"""
Sample year_predict query timing on GPU for training the GPU load predictor.

Runs predict_batch_float8('year_predict', 'gpu:0', ...) and records GPU metrics
alongside execution time into year_predict_gpu_load_training_data.
"""
import os
import time
import subprocess
import psycopg2

ALLOWED_CPUS = list(range(20))
ALLOWED_CPU_MASK = ",".join(str(c) for c in ALLOWED_CPUS)
DB_CONFIG = {
    "host": "localhost", "port": "5432", "user": "why",
    "dbname": "postgres", "password": "123456",
}

ROWS_PER_QUERY = 128
QUERIES_TOTAL = 30

# year_predict model metadata: shape 1x90, single Linear(90,1) layer
MODEL_MAC_COUNT = 90
MODEL_PARAM_COUNT = 91
MODEL_PARAM_SIZE = 364
DATA_SHAPE_1 = 1
DATA_SHAPE_2 = 90
DATA_SHAPE_3 = 0
DATA_SHAPE_4 = 0

YEAR_PREDICT_QUERY = (
    "SELECT predict_batch_float8('year_predict', 'gpu:0', data) "
    "OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) "
    "FROM year_predict_test LIMIT {row_count};"
)


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def get_backend_pid_and_pin(cur):
    cur.execute("SELECT pg_backend_pid()")
    backend_pid = cur.fetchone()[0]
    subprocess.run(
        ["taskset", "-p", "-c", ALLOWED_CPU_MASK, str(backend_pid)],
        capture_output=True, timeout=5
    )
    return backend_pid


def get_gpu_metrics():
    """Query nvidia-smi for GPU 0 metrics."""
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

        # Determine cuda_cores from compute capability
        cap_result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        cap = cap_result.stdout.strip()
        cuda_cores_map = {"8.0": 6912, "7.5": 4608, "7.0": 5120, "8.6": 10496}
        cuda_cores = float(cuda_cores_map.get(cap, 2048))

        return cuda_cores, gpu_freq, util, mem_used, mem_total
    except Exception as e:
        print(f"  Warning: nvidia-smi failed: {e}, using defaults")
        return 2048.0, 1500.0, 0.0, 0.0, 0.0


def run_query_and_collect(conn, row_count):
    cur = conn.cursor()
    get_backend_pid_and_pin(cur)
    cur.execute("SELECT register_process()")
    cur.execute("SET statement_timeout = '120s'")

    cuda_cores, gpu_freq, util, mem_used, mem_total = get_gpu_metrics()

    query = YEAR_PREDICT_QUERY.format(row_count=row_count)
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
            "INSERT INTO year_predict_gpu_load_training_data "
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
    cur.execute("SELECT count(*) FROM year_predict_gpu_load_training_data")
    count = cur.fetchone()[0]
    conn.close()
    return count


def main():
    os.sched_setaffinity(0, ALLOWED_CPUS)
    print("=" * 60)
    print("  year_predict GPU Training Data Collection")
    print("=" * 60)
    print(f"  Total queries: {QUERIES_TOTAL}")
    print(f"  Rows per query: {ROWS_PER_QUERY}")
    print()

    initial_count = get_training_count()
    print(f"  Initial training data count: {initial_count}")
    print()

    conn = get_conn()
    for i in range(QUERIES_TOTAL):
        r = run_query_and_collect(conn, ROWS_PER_QUERY)
        if r['status'] == 'success':
            print(f"  Query {i+1:2d}: {r['latency_ms']:.1f}ms | "
                  f"util={r['util']:.0f}% mem={r['mem_used']:.0f}MB | "
                  f"{r['windows']} windows inserted")
        else:
            print(f"  Query {i+1:2d}: ERROR - {r.get('error', 'unknown')}")
        time.sleep(0.3)
    conn.close()

    final_count = get_training_count()
    new_count = final_count - initial_count
    print(f"\n{'='*60}")
    print(f"  COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  New rows collected: {new_count}")
    print(f"  Total rows: {final_count}")

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            count(*) as rows,
            round(avg(execution_runtime_us)::numeric, 0) as avg_us,
            round(min(execution_runtime_us)::numeric, 0) as min_us,
            round(max(execution_runtime_us)::numeric, 0) as max_us,
            round(avg(util)::numeric, 1) as avg_util
        FROM year_predict_gpu_load_training_data
    """)
    row = cur.fetchone()
    print(f"\n  Summary: rows={row[0]}, avg={row[1]}us, min={row[2]}us, "
          f"max={row[3]}us, avg_util={row[4]}%")
    conn.close()


if __name__ == "__main__":
    main()
