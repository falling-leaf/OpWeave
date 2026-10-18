#!/usr/bin/env python3
"""
Hermes Baseline: YearPredict Dataset Execution Script.

Loads year_predict.pt model on CPU, fetches data from year_predict_test,
runs batched inference. Model loading excluded from timing; data loading
and inference both included (consistent with hermes_slice.py design).

Usage:
    python hermes_year_predict.py --row-count 10000 --json-output
    python hermes_year_predict.py --row-count 10000 --repetitions 3
"""
import os
import sys
import time
import json
import argparse
import torch
import numpy as np
import psycopg2

MODEL_PATH = "/home/why/baseline/models/year_predict.pt"
DB_CONFIG = {
    "host": "localhost", "port": "5432", "user": "why",
    "password": "123456", "dbname": "postgres",
}
BATCH_SIZE = 32


def load_data(n):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT get_mvec_data(data) FROM year_predict_test LIMIT %s", (n,))
    rows = cur.fetchall()
    conn.close()
    return [np.array(row[0], dtype=np.float32) for row in rows]


def run_inference(data_tensor, model):
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(data_tensor), BATCH_SIZE):
            batch = data_tensor[i:i + BATCH_SIZE]
            model(batch)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-count", type=int, default=10000)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--json-output", action="store_true")
    args = parser.parse_args()

    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()

    # Load data + build tensor (timed as part of execution per hermes_slice.py convention)
    t0 = time.perf_counter()
    data = load_data(args.row_count)
    data_tensor = torch.tensor(np.array(data), dtype=torch.float32)
    load_time = time.perf_counter() - t0

    # Warmup
    run_inference(data_tensor, model)

    times = []
    for _ in range(args.repetitions):
        t = run_inference(data_tensor, model)
        times.append(t)

    result = {
        "row_count": args.row_count,
        "repetitions": args.repetitions,
        "load_s": round(load_time, 4),
        "times": times,
        "avg": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
    }

    if args.json_output:
        print(json.dumps(result))
    else:
        print(f"Row count: {args.row_count}")
        for i, t in enumerate(times):
            print(f"  Run {i+1}: {t*1000:.1f}ms")
        print(f"  Avg: {result['avg']*1000:.1f}ms")


if __name__ == "__main__":
    main()
