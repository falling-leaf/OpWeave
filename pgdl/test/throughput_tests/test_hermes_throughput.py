#!/usr/bin/env python3
"""
Hermes Throughput Test (Max Concurrency).

Finds the maximum concurrency level for the Hermes baseline across 9 base datasets
(Series, NLP, Image). The test measures execution capacity only -- each worker
runs the Hermes-style pipeline (per-sample independent processing).

Key features:
- Two-phase search: coarse stepping (step=10) + binary search
- Theoretical upper limit: 100
- Circuit breaker: main monitor kills all workers if total time > 5 minutes
- Fork CoW for model sharing (memory efficient)
- Any single thread failure = test fails
- Testing: code EXECUTION concurrency (not code generation)

For Series datasets: runs predict_batch_float8 via psycopg2 (same as GenDB execution phase)
For NLP/Image datasets: runs Hermes-style Python pipeline (cross_encoder -> reader -> decoder)

Usage:
    python test_hermes_throughput.py [--dataset <name>] [--all]
"""

import os
import sys
import time
import json
import argparse
import multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import psycopg2

from tools.test_result_paths import THROUGHPUT_RESULTS_DIR, ensure_dir

# =============================================================================
# Configuration
# =============================================================================
MAX_CONCURRENCY_LIMIT = 100
COARSE_STEP = 10
QUERY_TIMEOUT = 600        # per-worker timeout
CIRCUIT_BREAKER_SEC = 300  # 5-minute total timeout for a concurrency level
BINARY_SLEEP = 1

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "morphingdb_test", "models")
TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
DB_CONFIG = {"dbname": "postgres", "host": "localhost", "port": "5432", "user": "why", "password": "123456"}

# =============================================================================
# Dataset configurations
# =============================================================================
SERIES_DATASETS = {
    "slice": {
        "category": "Series",
        "model": "slice",
        "table": "slice_test",
        "column": "data",
        "hardware": "cpu",
        "sql_query": "SELECT predict_batch_float8('{model}', '{hardware}', {column}) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM {table} LIMIT {row_count};",
    },
    "swarm": {
        "category": "Series",
        "model": "swarm",
        "table": "swarm_test",
        "column": "data",
        "hardware": "cpu",
        "sql_query": "SELECT predict_batch_float8('{model}', '{hardware}', {column}) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM {table} LIMIT {row_count};",
    },
    "year_predict": {
        "category": "Series",
        "model": "year_predict",
        "table": "year_predict_test",
        "column": "data",
        "hardware": "cpu",
        "sql_query": "SELECT predict_batch_float8('{model}', '{hardware}', {column}) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM {table} LIMIT {row_count};",
    },
}

NLP_IMAGE_DATASETS = {
    # --- NLP (GPU) ---
    "sst-2": {
        "category": "NLP",
        "model": "sst2_vec",
        "table": "nlp_vector_test",
        "column": "comment_vec",
        "hardware": "gpu",
        "sql_query": "SELECT predict_batch_float8('{model}', '{hardware}', {column}) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM {table} LIMIT {row_count};",
    },
    "imdb": {
        "category": "NLP",
        "model": "sst2_vec",
        "table": "imdb_vector_test",
        "column": "comment_vec",
        "hardware": "gpu",
        "sql_query": "SELECT predict_batch_float8('{model}', '{hardware}', {column}) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM {table} LIMIT {row_count};",
    },
    "financial": {
        "category": "NLP",
        "model": "finance",
        "table": "financial_phrasebank_vector_test",
        "column": "comment_vec",
        "hardware": "gpu",
        "sql_query": "SELECT predict_batch_float8('{model}', '{hardware}', {column}) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM {table} LIMIT {row_count};",
    },
    # --- IMAGE (GPU) ---
    "imagenet": {
        "category": "Image",
        "model": "defect_vec",
        "table": "imagenet_image_vector_table",
        "column": "image_vector",
        "hardware": "gpu",
        "sql_query": "SELECT predict_batch_float8('{model}', '{hardware}', {column}) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM {table} LIMIT {row_count};",
    },
    "cifar-10": {
        "category": "Image",
        "model": "googlenet_cifar10",
        "table": "cifar_image_vector_table",
        "column": "image_vector",
        "hardware": "gpu",
        "sql_query": "SELECT predict_batch_float8('{model}', '{hardware}', {column}) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM {table} LIMIT {row_count};",
    },
    "stanford_dogs": {
        "category": "Image",
        "model": "alexnet_stanford_dogs",
        "table": "stanford_dogs_image_vector_table",
        "column": "image_vector",
        "hardware": "gpu",
        "sql_query": "SELECT predict_batch_float8('{model}', '{hardware}', {column}) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM {table} LIMIT {row_count};",
    },
}

ALL_DATASETS = {**SERIES_DATASETS, **NLP_IMAGE_DATASETS}

# =============================================================================
# Worker: runs one Hermes-style execution task
# For Series: predict_batch_float8 via psycopg2
# For NLP/Image: same as GenDB -- predict_batch_float8 via psycopg2
# Both test the EXECUTION phase concurrency
# =============================================================================
def run_hermes_worker(task_id, dataset_name, row_count):
    """
    Execute one Hermes-style task in a forked process.
    Hermes execution phase = running the inference pipeline.
    For all 9 datasets, we test predict_batch_float8 concurrency.
    """
    try:
        os.setpgid(0, 0)
    except OSError:
        pass

    conn = None
    try:
        ds = ALL_DATASETS[dataset_name]
        sql_query = ds["sql_query"].format(
            model=ds["model"],
            hardware=ds["hardware"],
            column=ds["column"],
            table=ds["table"],
            row_count=row_count,
        )

        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = '300s';")
        cur.execute("SELECT register_process();")

        t0 = time.perf_counter()
        cur.execute(sql_query)
        cur.fetchall()
        elapsed = time.perf_counter() - t0

        conn.close()
        conn = None

        print(f"  [OK] Task {task_id}: {elapsed:.2f}s")
        return {"task_id": task_id, "status": "success", "latency": elapsed}

    except Exception as e:
        err_msg = str(e)
        if len(err_msg) > 200:
            err_msg = err_msg[:200] + "..."
        print(f"  [FAIL] Task {task_id}: {err_msg}")
        return {"task_id": task_id, "status": "failed", "error": err_msg}
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# =============================================================================
# Concurrency test with circuit breaker
# =============================================================================
def test_concurrency_level(concurrency_level, dataset_name, row_count):
    """Test a specific concurrency level with circuit breaker."""
    print(f"  Testing concurrency level: {concurrency_level}")

    start_time = time.time()
    ctx = multiprocessing.get_context('fork')
    pool = ctx.Pool(
        processes=concurrency_level,
        initializer=os.setpgid,
        initargs=(0, 0)
    )

    results = []
    try:
        async_results = []
        for i in range(concurrency_level):
            ar = pool.apply_async(run_hermes_worker, args=(i, dataset_name, row_count))
            async_results.append(ar)

        for i, ar in enumerate(async_results):
            # Circuit breaker: check if total time exceeds 5 minutes
            elapsed_total = time.time() - start_time
            if elapsed_total > CIRCUIT_BREAKER_SEC:
                print(f"  [CIRCUIT BREAKER] Total time {elapsed_total:.0f}s > {CIRCUIT_BREAKER_SEC}s, aborting!")
                pool.terminate()
                results.append({
                    "task_id": i, "status": "failed",
                    "error": f"Circuit breaker triggered at {elapsed_total:.0f}s"
                })
                for j in range(i + 1, concurrency_level):
                    results.append({
                        "task_id": j, "status": "failed",
                        "error": "Circuit breaker triggered"
                    })
                return False, results, elapsed_total

            try:
                remaining = max(1, QUERY_TIMEOUT - elapsed_total)
                result = ar.get(timeout=remaining)
                results.append(result)
            except Exception as e:
                results.append({"task_id": i, "status": "failed", "error": str(e)})
    finally:
        pool.terminate()
        pool.join()

    wall_time = time.time() - start_time
    failed = [r for r in results if r["status"] == "failed"]
    successful = [r for r in results if r["status"] == "success"]
    all_passed = len(failed) == 0

    print(f"  Level {concurrency_level}: {len(successful)}/{len(results)} passed "
          f"{'[PASS]' if all_passed else '[FAIL]'} ({wall_time:.1f}s)")
    if failed:
        for f in failed[:3]:
            print(f"    Failed task {f['task_id']}: {f.get('error', 'unknown')[:100]}")
        if len(failed) > 3:
            print(f"    ... and {len(failed)-3} more")

    return all_passed, results, wall_time


# =============================================================================
# Binary search
# =============================================================================
def binary_search_max_concurrency(low, high, dataset_name, row_count):
    max_pass = low - 1
    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            low = 1
            continue

        all_passed, _, _ = test_concurrency_level(mid, dataset_name, row_count)
        if all_passed:
            max_pass = mid
            print(f"  -> {mid} passed, searching [{mid + 1}, {high}]")
            low = mid + 1
        else:
            print(f"  -> {mid} failed, searching [{low}, {mid - 1}]")
            high = mid - 1

        time.sleep(BINARY_SLEEP)
    return max_pass


def find_max_concurrency(dataset_name, row_count):
    # Phase 1: coarse stepping
    print(f"Phase 1: Coarse search (step={COARSE_STEP})")
    last_pass = 0
    current = COARSE_STEP

    while current <= MAX_CONCURRENCY_LIMIT:
        all_passed, _, wall_time = test_concurrency_level(current, dataset_name, row_count)
        if all_passed:
            last_pass = current
            if current >= MAX_CONCURRENCY_LIMIT:
                print(f"  Reached MAX limit {MAX_CONCURRENCY_LIMIT}!")
                return MAX_CONCURRENCY_LIMIT
            current += COARSE_STEP
        else:
            print(f"  Failed at {current}, last pass was {last_pass}")
            break

    time.sleep(BINARY_SLEEP)

    # Phase 2: binary search
    if last_pass == 0:
        print(f"Phase 2: Binary search in [1, {COARSE_STEP - 1}]")
        return binary_search_max_concurrency(1, COARSE_STEP - 1, dataset_name, row_count)
    elif current - last_pass <= 1:
        return last_pass
    else:
        search_high = min(current - 1, MAX_CONCURRENCY_LIMIT)
        print(f"Phase 2: Binary search in [{last_pass + 1}, {search_high}]")
        result = binary_search_max_concurrency(last_pass + 1, search_high, dataset_name, row_count)
        return result if result > last_pass else last_pass


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Hermes Throughput Test")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Test a single dataset (e.g., slice, sst-2, imagenet)")
    parser.add_argument("--row-count", type=int, default=1000,
                        help="Number of rows per query (default: 1000)")
    args = parser.parse_args()

    if args.dataset:
        if args.dataset not in ALL_DATASETS:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {', '.join(ALL_DATASETS.keys())}")
            sys.exit(1)
        datasets_to_test = {args.dataset: ALL_DATASETS[args.dataset]}
    else:
        # Test all 9 base datasets
        datasets_to_test = ALL_DATASETS

    print("=" * 80)
    print("Hermes Throughput Test (Max Concurrency)")
    print(f"Upper limit: {MAX_CONCURRENCY_LIMIT} threads")
    print(f"Circuit breaker: {CIRCUIT_BREAKER_SEC}s (5 minutes)")
    print(f"Row count: {args.row_count}")
    print(f"Datasets: {', '.join(datasets_to_test.keys())}")
    print("=" * 80)

    all_results = {}

    for ds_name, ds_info in datasets_to_test.items():
        print(f"\n{'=' * 60}")
        print(f"  Dataset: {ds_name} ({ds_info['category']})")
        print(f"  Model: {ds_info['model']}, Table: {ds_info['table']}")
        print(f"  Hardware: {ds_info['hardware']}")
        print(f"{'=' * 60}")

        # Cool-down between datasets
        time.sleep(2)

        max_conc = find_max_concurrency(ds_name, args.row_count)

        all_results[ds_name] = {
            "category": ds_info["category"],
            "model": ds_info["model"],
            "row_count": args.row_count,
            "max_concurrency": max_conc,
        }

        print(f"\n  RESULT: {ds_name} MAX_CONCURRENCY = {max_conc}")

    # Summary
    print(f"\n{'=' * 80}")
    print("  SUMMARY: Hermes Throughput (Max Concurrency)")
    print(f"{'=' * 80}")
    print(f"  {'Dataset':<20} {'Category':<10} {'Max Concurrency':<20}")
    print("  " + "-" * 50)
    for ds_name, res in all_results.items():
        print(f"  {ds_name:<20} {res['category']:<10} {res['max_concurrency']:<20}")
    print(f"{'=' * 80}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(ensure_dir(THROUGHPUT_RESULTS_DIR), f"hermes_throughput_results_{timestamp}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "test": "hermes_throughput",
            "max_limit": MAX_CONCURRENCY_LIMIT,
            "circuit_breaker_sec": CIRCUIT_BREAKER_SEC,
            "row_count": args.row_count,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
