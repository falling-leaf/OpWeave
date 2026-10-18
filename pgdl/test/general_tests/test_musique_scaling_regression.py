#!/usr/bin/env python3
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import psycopg2

from morphingdb_test.config import db_config

SQL_TEMPLATE = (
    "SELECT unnest(db_agent_single('musique', sub_table.reasoning_vec)) AS score "
    "FROM (SELECT * FROM musique_vector_test ORDER BY id LIMIT {n_samples}) AS sub_table;"
)

def run_query_with_cursor(cur, n_samples):
    start = time.time()
    cur.execute(SQL_TEMPLATE.format(n_samples=n_samples))
    rows = cur.fetchall()
    elapsed = time.time() - start
    return elapsed, len(rows)


def run_query(n_samples):
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute("select register_process();")
    elapsed, row_count = run_query_with_cursor(cur, n_samples)
    conn.close()
    return elapsed, row_count


def run_persistent_probe():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute("select register_process();")
    warm_latency, warm_rows = run_query_with_cursor(cur, 1)
    hot_latency, hot_rows = run_query_with_cursor(cur, 1)
    lat_10, rows_10 = run_query_with_cursor(cur, 10)
    conn.close()
    return warm_latency, warm_rows, hot_latency, hot_rows, lat_10, rows_10


def main():
    # Fresh-connection path preserves the canonical scaling semantics.
    warm_latency, warm_rows = run_query(1)
    lat_1, rows_1 = run_query(1)
    lat_10, rows_10 = run_query(10)

    print(f"fresh warmup n=1 latency={warm_latency:.4f}s rows={warm_rows}")
    print(f"fresh measure n=1 latency={lat_1:.4f}s rows={rows_1}")
    print(f"fresh measure n=10 latency={lat_10:.4f}s rows={rows_10}")

    assert warm_rows == 1, warm_rows
    assert rows_1 == 1, rows_1
    assert rows_10 == 10, rows_10
    assert lat_10 > lat_1 * 1.2, (lat_1, lat_10)

    # Persistent-connection path validates backend-local hot-load reuse.
    persistent_warm, persistent_warm_rows, persistent_hot, persistent_hot_rows, persistent_10, persistent_10_rows = run_persistent_probe()
    print(f"persistent warmup n=1 latency={persistent_warm:.4f}s rows={persistent_warm_rows}")
    print(f"persistent hot n=1 latency={persistent_hot:.4f}s rows={persistent_hot_rows}")
    print(f"persistent n=10 latency={persistent_10:.4f}s rows={persistent_10_rows}")

    assert persistent_warm_rows == 1, persistent_warm_rows
    assert persistent_hot_rows == 1, persistent_hot_rows
    assert persistent_10_rows == 10, persistent_10_rows
    assert persistent_hot < persistent_warm * 0.85, (persistent_warm, persistent_hot)
    assert persistent_10 > persistent_hot * 1.2, (persistent_hot, persistent_10)


if __name__ == "__main__":
    main()
