#!/usr/bin/env python3
"""
Sample reasoning (cross_encoder) query timing on CPU at different CPU load
levels for training the CPU load predictor.

For the HotpotQA reasoning task, the cross_encoder stage goes through the
OptimizationAgent scheduler. We collect CPU inference timing under various
CPU load conditions so DBAgent can estimate the CPU cost.

Load levels use percentage-based spin workers on CPUs 0-19 (same pattern as
year_predict / cifar10).

Data shape: (1, 4, 128)  -- reasoning_vec mvec shape
Model: cross_encoder (22.7M params, ~91MB)
"""
import os
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

LOAD_LEVELS = [10, 30, 50, 70, 90]
ROW_COUNT = 20          # all available reasoning_vector_test rows
WINDOW = 16             # OVER window: ROWS BETWEEN CURRENT ROW AND 15 FOLLOWING
QUERIES_PER_LEVEL = 20  # repeat to accumulate ~100 samples

# cross_encoder: input shape (1,4,128)
# metadata computed via torch (param_count, param_size, linear-MAC)
MODEL_MAC_COUNT = 22682496
MODEL_PARAM_COUNT = 22713601
MODEL_PARAM_SIZE = 90854404
DATA_SHAPE_1 = 1
DATA_SHAPE_2 = 4
DATA_SHAPE_3 = 128
DATA_SHAPE_4 = 0

REASONING_CPU_QUERY = (
    "SELECT predict_batch_float8('cross_encoder', 'cpu', reasoning_vec) "
    "OVER (ROWS BETWEEN CURRENT ROW AND {win} FOLLOWING) "
    "FROM reasoning_vector_test LIMIT {row_count};"
)


def _spin_worker(stop_flag):
    os.sched_setaffinity(0, ALLOWED_CPUS)
    while not stop_flag.value:
        pass


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def get_cpu_load():
    with open('/proc/loadavg') as f:
        return float(f.read().strip().split()[0])


def get_backend_pid_and_pin(cur):
    cur.execute("SELECT pg_backend_pid()")
    pid = cur.fetchone()[0]
    subprocess.run(
        ["taskset", "-p", "-c", ALLOWED_CPU_MASK, str(pid)],
        capture_output=True, timeout=5
    )


def run_query_and_collect(conn, cpu_load_snapshot):
    cur = conn.cursor()
    get_backend_pid_and_pin(cur)
    cur.execute("SELECT register_process()")
    cur.execute("SET statement_timeout = '300s'")

    query = REASONING_CPU_QUERY.format(win=WINDOW - 1, row_count=ROW_COUNT)
    start = time.perf_counter()
    try:
        cur.execute(query)
        cur.fetchall()
    except Exception as e:
        conn.rollback()
        return {'status': 'error', 'error': str(e), 'latency_ms': 0}

    elapsed_us = int((time.perf_counter() - start) * 1e6)
    cpu_cores = 40

    cur.execute(
        "INSERT INTO reasoning_cpu_load_training_data "
        "(cpu_load, cpu_cores, execution_runtime_us, "
        " data_shape_1, data_shape_2, data_shape_3, data_shape_4, "
        " model_mac_count, model_param_count, model_param_size) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (cpu_load_snapshot, cpu_cores, elapsed_us,
         DATA_SHAPE_1, DATA_SHAPE_2, DATA_SHAPE_3, DATA_SHAPE_4,
         MODEL_MAC_COUNT, MODEL_PARAM_COUNT, MODEL_PARAM_SIZE)
    )
    conn.commit()
    return {'status': 'success', 'latency_ms': round(elapsed_us / 1000, 1)}


def get_training_count():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM reasoning_cpu_load_training_data")
    count = cur.fetchone()[0]
    conn.close()
    return count


def main():
    os.sched_setaffinity(0, ALLOWED_CPUS)
    print("=" * 60)
    print("  reasoning (cross_encoder) CPU Training Data Collection")
    print("=" * 60)
    print(f"  Load levels: {LOAD_LEVELS}%")
    print(f"  Queries per level: {QUERIES_PER_LEVEL}")
    print(f"  Rows per query: {ROW_COUNT} (window {WINDOW})")
    print()

    initial_count = get_training_count()
    print(f"  Initial count: {initial_count}")

    ctx = multiprocessing.get_context('fork')

    for load_pct in LOAD_LEVELS:
        print(f"\n{'='*60}")
        print(f"  LOAD LEVEL: {load_pct}%")
        print(f"{'='*60}")

        n_workers = max(1, int(len(ALLOWED_CPUS) * load_pct / 100))
        print(f"  Spin workers: {n_workers} on {len(ALLOWED_CPUS)} CPUs")

        stop_flag = ctx.Value('b', False)
        procs = []
        for _ in range(n_workers):
            p = ctx.Process(target=_spin_worker, args=(stop_flag,))
            p.start()
            procs.append(p)

        print("  Stabilizing for 3s...")
        time.sleep(3)

        cpu_load_snapshot = get_cpu_load()
        print(f"  Load average (1min): {cpu_load_snapshot:.2f}")

        conn = get_conn()
        for i in range(QUERIES_PER_LEVEL):
            r = run_query_and_collect(conn, cpu_load_snapshot)
            if r['status'] == 'success':
                print(f"    Query {i+1:2d}: {r['latency_ms']:.1f}ms")
            else:
                print(f"    Query {i+1:2d}: ERROR - {r.get('error', 'unknown')}")
            time.sleep(0.3)
        conn.close()

        stop_flag.value = True
        for p in procs:
            p.kill()
        for p in procs:
            p.join(timeout=1)

        print("  Cooling down for 8s...")
        time.sleep(8)

        current_count = get_training_count()
        print(f"  Count after this level: {current_count}")

    final_count = get_training_count()
    print(f"\n{'='*60}")
    print(f"  COLLECTION COMPLETE: {final_count - initial_count} new, {final_count} total")
    print(f"{'='*60}")

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT round(cpu_load::numeric, 1) as load_bucket, count(*) as rows,
               round(avg(execution_runtime_us)::numeric, 0) as avg_us,
               round(min(execution_runtime_us)::numeric, 0) as min_us,
               round(max(execution_runtime_us)::numeric, 0) as max_us
        FROM reasoning_cpu_load_training_data GROUP BY 1 ORDER BY 1
    """)
    print(f"\n  {'Load':>6s} | {'Rows':>5s} | {'Avg_us':>10s} | {'Min_us':>8s} | {'Max_us':>8s}")
    for row in cur.fetchall():
        print(f"  {row[0]:>6.1f} | {row[1]:5d} | {row[2]:>10.0f} | {row[3]:>8.0f} | {row[4]:>8.0f}")
    conn.close()


if __name__ == "__main__":
    main()
