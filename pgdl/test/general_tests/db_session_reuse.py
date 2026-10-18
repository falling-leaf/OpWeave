#!/usr/bin/env python3
"""
DBAgent & MorphingDB Session Reuse Benchmark on Series Datasets
Measures cumulative latency of repeated SQL queries within a single
PostgreSQL session.

Datasets: slice, swarm, year_predict
Query counts: 1, 5, 10, 50, 100
Rows per query: 1000

Methodology:
  - One PostgreSQL connection per dataset
  - For each query count q, run the same SQL query q times sequentially
  - Measure cumulative wall-clock time
  - 3 timed runs per query count, report min
  - Warmup query before timing
"""

import os
import sys
import time
import json
import psycopg2
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from morphingdb_test.config import db_config

ROWS_PER_QUERY = 1000
QUERY_COUNTS = [1, 5, 10, 50, 100]
NUM_RUNS = 3

DATASETS = {
    'slice': {
        'table': 'slice_test',
        'column': 'data',
        'model': 'slice',
        'func_type': 'series',
        'symbol': 'cpu',
    },
    'swarm': {
        'table': 'swarm_test',
        'column': 'data',
        'model': 'swarm',
        'func_type': 'series',
        'symbol': 'cpu',
    },
    'year_predict': {
        'table': 'year_predict_test',
        'column': 'data',
        'model': 'year_predict',
        'func_type': 'series',
        'symbol': 'cpu',
    },
}

from support.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir

OUTPUT_DIR = ensure_dir(GENERAL_RESULTS_DIR)


def dbagent_query(ds, n):
    return (
        "SELECT unnest(db_agent_single('{func_type}', sub_table.{column})) AS score "
        "FROM (SELECT * FROM {table} LIMIT {n}) AS sub_table;"
    ).format(n=n, **ds)


def morphingdb_query(ds, n):
    return (
        "SELECT predict_batch_float8('{model}', '{symbol}', {column}) "
        "OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) "
        "FROM {table} LIMIT {n};"
    ).format(n=n, **ds)


def benchmark_system(label, query_fn):
    """Run session reuse benchmark for one system."""
    print(f"\n{'='*60}")
    print(f"Session Reuse Benchmark: {label}")
    print(f"{'='*60}")

    all_results = {}

    for ds_name, ds in DATASETS.items():
        print(f"\n  Dataset: {ds_name}")
        sql = query_fn(ds, ROWS_PER_QUERY)

        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute("SELECT register_process();")

        # Warmup
        print(f"    Warmup...")
        cur.execute(sql)
        cur.fetchall()

        ds_results = {}
        for q in QUERY_COUNTS:
            print(f"    Query count = {q}:")
            times = []
            for run in range(NUM_RUNS):
                t0 = time.perf_counter()
                for _ in range(q):
                    cur.execute(sql)
                    rows = cur.fetchall()
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                print(f"      Run {run+1}/{NUM_RUNS}: {elapsed:.4f}s")

            min_time = min(times)
            ds_results[q] = {
                "cumulative_s": round(min_time, 4),
                "per_query_ms": round(min_time / q * 1000, 2),
                "all_runs_s": [round(t, 4) for t in times],
            }
            print(f"      Min cumulative: {min_time:.4f}s ({min_time/q*1000:.2f} ms/query)")

        conn.close()
        all_results[ds_name] = ds_results

    return all_results


def main():
    print("=" * 70)
    print("  DBAGENT & MORPHINGDB SESSION REUSE BENCHMARK")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"  Rows per query: {ROWS_PER_QUERY}")
    print(f"  Query counts: {QUERY_COUNTS}")
    print(f"  Runs per count: {NUM_RUNS} (min reported)")
    print("=" * 70)

    all_results = {
        "experiment": "db_session_reuse",
        "timestamp": datetime.now().isoformat(),
        "rows_per_query": ROWS_PER_QUERY,
        "query_counts": QUERY_COUNTS,
        "num_runs": NUM_RUNS,
        "systems": {},
    }

    # DBAgent
    try:
        r = benchmark_system("DBAgent", dbagent_query)
        all_results["systems"]["dbagent"] = r
    except Exception as e:
        print(f"ERROR (DBAgent): {e}")
        import traceback; traceback.print_exc()
        all_results["systems"]["dbagent"] = {"error": str(e)}

    # MorphingDB
    try:
        r = benchmark_system("MorphingDB", morphingdb_query)
        all_results["systems"]["morphingdb"] = r
    except Exception as e:
        print(f"ERROR (MorphingDB): {e}")
        import traceback; traceback.print_exc()
        all_results["systems"]["morphingdb"] = {"error": str(e)}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(
        OUTPUT_DIR,
        f"db_session_reuse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Summary
    for sys_name in ["dbagent", "morphingdb"]:
        sys_data = all_results["systems"].get(sys_name, {})
        if "error" in sys_data:
            print(f"\n{sys_name}: ERROR - {sys_data['error']}")
            continue
        print(f"\n{'='*60}")
        print(f"  {sys_name}: Cumulative Latency (s)")
        print(f"{'='*60}")
        header = f"{'Dataset':<18}"
        for q in QUERY_COUNTS:
            header += f" {'q=' + str(q):>12}"
        print(header)
        print("-" * (18 + 13 * len(QUERY_COUNTS)))
        for ds_name in DATASETS:
            ds_results = sys_data.get(ds_name, {})
            if not ds_results:
                continue
            row = f"{ds_name:<18}"
            for q in QUERY_COUNTS:
                v = ds_results.get(q, {}).get("cumulative_s", "N/A")
                if isinstance(v, float):
                    row += f" {v:>12.4f}"
                else:
                    row += f" {'N/A':>12}"
            print(row)

    print("\nDone!")


if __name__ == "__main__":
    main()
