#!/usr/bin/env python3
"""
Sample slice query data at different CPU load levels for training the CPU load predictor.

This script:
1. Uses fork spin workers to generate different CPU load levels
2. Runs slice queries to trigger data collection (via SLICE_BATCH_SAMPLE_BUTTON in batch_interface.cpp)
3. Commits after each query so SPI_execute inserts persist

Usage:
    python sample_slice_training_data.py [--load-levels 10,30,50,70,90] [--reps 3]
"""
import os
import sys
import time
import subprocess
import multiprocessing
import psycopg2

# =============================================================================
# Configuration
# =============================================================================
ALLOWED_CPUS = list(range(20))
ALLOWED_CPU_MASK = ",".join(str(c) for c in ALLOWED_CPUS)
DB_CONFIG = {
    "host": "localhost", "port": "5432", "user": "why",
    "dbname": "postgres", "password": "123456",
}

# Load levels to sample (percentage)
LOAD_LEVELS = [10, 30, 50, 70, 90]
ROWS_PER_QUERY = 128  # Need >= WINDOW_SIZE (32) rows per query
QUERIES_PER_LEVEL = 6  # Each query inserts 4 data points (4 batches of 32 in 128 rows)

SLICE_QUERY = (
    "SELECT predict_batch_float8('slice', 'cpu', data) "
    "OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) "
    "FROM slice_test LIMIT {row_count};"
)


def _spin_worker(stop_flag):
    """Spin worker on allowed CPUs."""
    os.sched_setaffinity(0, ALLOWED_CPUS)
    while not stop_flag.value:
        pass


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def get_backend_pid_and_pin(cur):
    """Get PostgreSQL backend PID and pin it to allowed CPUs."""
    cur.execute("SELECT pg_backend_pid()")
    backend_pid = cur.fetchone()[0]
    subprocess.run(
        ["taskset", "-p", "-c", ALLOWED_CPU_MASK, str(backend_pid)],
        capture_output=True, timeout=5
    )
    return backend_pid


def run_slice_query(conn, row_count):
    """Run a slice query and return the result."""
    cur = conn.cursor()
    
    # Get backend PID and pin it
    get_backend_pid_and_pin(cur)
    
    # Register process (required by predict_batch_float8)
    cur.execute("SELECT register_process()")
    
    # Execute query
    query = SLICE_QUERY.format(row_count=row_count)
    cur.execute("SET statement_timeout = '120s'")
    
    start = time.perf_counter()
    try:
        cur.execute(query)
        try:
            cur.fetchall()
        except Exception:
            pass
        # CRITICAL: commit the transaction so SPI_execute inserts persist
        conn.commit()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {'latency_ms': round(elapsed_ms, 1), 'status': 'success'}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {'latency_ms': round(elapsed_ms, 1), 'status': 'error', 'error': str(e)}


def get_training_count():
    """Get current training data count."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM slice_cpu_load_training_data")
    count = cur.fetchone()[0]
    conn.close()
    return count


def main():
    print("=" * 60)
    print("  Slice Training Data Collection")
    print("=" * 60)
    print(f"  Load levels: {LOAD_LEVELS}%")
    print(f"  Queries per level: {QUERIES_PER_LEVEL}")
    print(f"  Rows per query: {ROWS_PER_QUERY}")
    print(f"  Allowed CPUs: {ALLOWED_CPU_MASK}")
    print()
    
    # Check initial training data count
    initial_count = get_training_count()
    print(f"  Initial training data count: {initial_count}")
    print()
    
    ctx = multiprocessing.get_context('fork')
    
    for load_pct in LOAD_LEVELS:
        print(f"\n{'='*60}")
        print(f"  LOAD LEVEL: {load_pct}%")
        print(f"{'='*60}")
        
        # Calculate number of workers for target load
        n_workers = int(len(ALLOWED_CPUS) * load_pct / 100)
        n_workers = max(1, n_workers)
        
        print(f"  Workers: {n_workers} on {len(ALLOWED_CPUS)} CPUs (target {load_pct}%)")
        
        # Start load workers
        stop_flag = ctx.Value('b', False)
        procs = []
        for _ in range(n_workers):
            p = ctx.Process(target=_spin_worker, args=(stop_flag,))
            p.start()
            procs.append(p)
        
        # Stabilize
        print(f"  Stabilizing for 3s...")
        time.sleep(3)
        
        # Check actual load
        with open('/proc/loadavg') as f:
            loadavg = f.read().strip().split()[0]
        print(f"  Load average: {loadavg}")
        
        # Run queries
        conn = get_conn()
        for i in range(QUERIES_PER_LEVEL):
            r = run_slice_query(conn, ROWS_PER_QUERY)
            status = "OK" if r["status"] == "success" else f"!! {r.get('error', 'unknown')}"
            print(f"    Query {i+1}: {r['latency_ms']:.1f}ms | {status}")
            time.sleep(0.5)
        conn.close()
        
        # Stop workers
        stop_flag.value = True
        for p in procs:
            p.kill()
        for p in procs:
            p.join(timeout=1)
        
        # Cooldown
        print(f"  Cooling down for 10s...")
        time.sleep(10)
        
        # Check training data count
        current_count = get_training_count()
        print(f"  Training data count after this level: {current_count}")
    
    # Final summary
    final_count = get_training_count()
    new_count = final_count - initial_count
    print(f"\n{'='*60}")
    print(f"  COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  New training data rows collected: {new_count}")
    print(f"  Total training data rows: {final_count}")
    
    # Show collected data summary by load level
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            CASE 
                WHEN cpu_load < 1.0 THEN '0-5%'
                WHEN cpu_load < 3.0 THEN '5-15%'
                WHEN cpu_load < 5.0 THEN '15-25%'
                WHEN cpu_load < 8.0 THEN '25-40%'
                WHEN cpu_load < 12.0 THEN '40-60%'
                ELSE '60%+'
            END as load_range,
            count(*) as rows,
            round(avg(cpu_load), 2) as avg_load,
            round(avg(execution_runtime_us), 0) as avg_runtime_us,
            round(min(execution_runtime_us), 0) as min_runtime_us,
            round(max(execution_runtime_us), 0) as max_runtime_us
        FROM slice_cpu_load_training_data
        GROUP BY 1 ORDER BY 1
    """)
    rows = cur.fetchall()
    print(f"\n  Data by load range:")
    print(f"  {'Load Range':>10s} | {'Rows':>5s} | {'AvgLoad':>8s} | {'Avg_us':>10s} | {'Min_us':>8s} | {'Max_us':>8s}")
    print(f"  {'-'*10} | {'-'*5} | {'-'*8} | {'-'*10} | {'-'*8} | {'-'*8}")
    for row in rows:
        print(f"  {row[0]:>10s} | {row[1]:5d} | {row[2]:>8.2f} | {row[3]:>10.0f} | {row[4]:>8.0f} | {row[5]:>8.0f}")
    conn.close()


if __name__ == "__main__":
    main()
