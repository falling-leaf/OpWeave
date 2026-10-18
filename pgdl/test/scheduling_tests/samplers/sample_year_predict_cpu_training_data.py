#!/usr/bin/env python3
"""
Sample year_predict query timing at different CPU load levels for training the CPU load predictor.

Collects timing data directly from Python (batch_interface.cpp only handles slice shape 1x384,
not year_predict shape 1x90), and inserts into year_predict_cpu_load_training_data.

Load levels use percentage-based spin workers on CPUs 0-19.
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
ROWS_PER_QUERY = 128  # 4 windows of WINDOW_SIZE=32
QUERIES_PER_LEVEL = 8

# year_predict model metadata: shape 1x90, single Linear(90,1) layer
MODEL_MAC_COUNT = 90      # 1 * 90
MODEL_PARAM_COUNT = 91    # 90 weights + 1 bias
MODEL_PARAM_SIZE = 364    # 91 * 4 bytes
DATA_SHAPE_1 = 1
DATA_SHAPE_2 = 90
DATA_SHAPE_3 = 0
DATA_SHAPE_4 = 0

YEAR_PREDICT_QUERY = (
    "SELECT predict_batch_float8('year_predict', 'cpu', data) "
    "OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) "
    "FROM year_predict_test LIMIT {row_count};"
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
    backend_pid = cur.fetchone()[0]
    subprocess.run(
        ["taskset", "-p", "-c", ALLOWED_CPU_MASK, str(backend_pid)],
        capture_output=True, timeout=5
    )
    return backend_pid


def run_query_and_collect(conn, row_count, cpu_load_snapshot):
    cur = conn.cursor()
    get_backend_pid_and_pin(cur)
    cur.execute("SELECT register_process()")
    cur.execute("SET statement_timeout = '120s'")

    query = YEAR_PREDICT_QUERY.format(row_count=row_count)
    start = time.perf_counter()
    try:
        cur.execute(query)
        cur.fetchall()
    except Exception as e:
        conn.rollback()
        return {'status': 'error', 'error': str(e), 'latency_ms': 0}

    elapsed_us = int((time.perf_counter() - start) * 1e6)

    # Insert one row per WINDOW_SIZE=32 window
    n_windows = row_count // 32
    per_window_us = elapsed_us // max(n_windows, 1)
    cpu_cores = len(ALLOWED_CPUS)

    for _ in range(n_windows):
        cur.execute(
            "INSERT INTO year_predict_cpu_load_training_data "
            "(cpu_load, cpu_cores, execution_runtime_us, "
            " data_shape_1, data_shape_2, data_shape_3, data_shape_4, "
            " model_mac_count, model_param_count, model_param_size) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (cpu_load_snapshot, cpu_cores, per_window_us,
             DATA_SHAPE_1, DATA_SHAPE_2, DATA_SHAPE_3, DATA_SHAPE_4,
             MODEL_MAC_COUNT, MODEL_PARAM_COUNT, MODEL_PARAM_SIZE)
        )

    conn.commit()
    return {'status': 'success', 'latency_ms': round(elapsed_us / 1000, 1), 'windows': n_windows}


def get_training_count():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM year_predict_cpu_load_training_data")
    count = cur.fetchone()[0]
    conn.close()
    return count


def main():
    os.sched_setaffinity(0, ALLOWED_CPUS)
    print("=" * 60)
    print("  year_predict CPU Training Data Collection")
    print("=" * 60)
    print(f"  Load levels: {LOAD_LEVELS}%")
    print(f"  Queries per level: {QUERIES_PER_LEVEL}")
    print(f"  Rows per query: {ROWS_PER_QUERY}")
    print(f"  Allowed CPUs: 0-{len(ALLOWED_CPUS)-1} ({len(ALLOWED_CPUS)} total)")
    print()

    initial_count = get_training_count()
    print(f"  Initial training data count: {initial_count}")
    print()

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
            r = run_query_and_collect(conn, ROWS_PER_QUERY, cpu_load_snapshot)
            if r['status'] == 'success':
                print(f"    Query {i+1}: {r['latency_ms']:.1f}ms | {r['windows']} windows inserted")
            else:
                print(f"    Query {i+1}: ERROR - {r.get('error', 'unknown')}")
            time.sleep(0.5)
        conn.close()

        stop_flag.value = True
        for p in procs:
            p.kill()
        for p in procs:
            p.join(timeout=1)

        print("  Cooling down for 10s...")
        time.sleep(10)

        current_count = get_training_count()
        print(f"  Training data count after this level: {current_count}")

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
            round(cpu_load::numeric, 0) as load_bucket,
            count(*) as rows,
            round(avg(execution_runtime_us)::numeric, 0) as avg_us,
            round(min(execution_runtime_us)::numeric, 0) as min_us,
            round(max(execution_runtime_us)::numeric, 0) as max_us
        FROM year_predict_cpu_load_training_data
        GROUP BY 1 ORDER BY 1
    """)
    rows = cur.fetchall()
    print(f"\n  Data by load bucket:")
    print(f"  {'Load':>6s} | {'Rows':>5s} | {'Avg_us':>10s} | {'Min_us':>8s} | {'Max_us':>8s}")
    print(f"  {'-'*6} | {'-'*5} | {'-'*10} | {'-'*8} | {'-'*8}")
    for row in rows:
        print(f"  {row[0]:>6.0f} | {row[1]:5d} | {row[2]:>10.0f} | {row[3]:>8.0f} | {row[4]:>8.0f}")
    conn.close()


if __name__ == "__main__":
    main()
