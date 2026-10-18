#!/usr/bin/env python3
"""
EvaDB Throughput Test (Max Concurrency).

Finds the maximum concurrency level for evadb across 8 base datasets.
The test measures execution capacity only -- each worker runs the same
evadb query concurrently.

Key features:
- Function created ONCE in main process, workers reuse it
- Two-phase search: coarse stepping (step=10) + binary search
- Theoretical upper limit: 100
- Circuit breaker: main monitor kills all workers if total time > 5 minutes
- Any single thread failure = test fails

Usage:
    python test_evadb_throughput_full.py [--dataset <name>] [--row-count 1000]
"""

import os
import sys
import time
import json
import argparse
import multiprocessing
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import evadb
import warnings
warnings.filterwarnings("ignore")

from tools.test_result_paths import THROUGHPUT_RESULTS_DIR, ensure_dir

# =============================================================================
# Configuration
# =============================================================================
MAX_CONCURRENCY_LIMIT = 100
COARSE_STEP = 10
QUERY_TIMEOUT = 600        # per-worker timeout
CIRCUIT_BREAKER_SEC = 300  # 5-minute total timeout for a concurrency level
BINARY_SLEEP = 1

DB_CONFIG = {
    "database": "postgres",
    "host": "localhost",
    "port": "5432",
    "user": "why",
    "password": "123456"
}

# =============================================================================
# Dataset configurations
# =============================================================================
ALL_DATASETS = {
    "slice": {
        "category": "Series",
        "func_name": "SliceFunction",
        "cols": ", ".join([f"value{i}" for i in range(384)]),
        "query_sql": "SELECT SliceFunction({cols}) FROM postgres_data.slice_origin_test LIMIT {n};",
    },
    "year_predict": {
        "category": "Series",
        "func_name": "YearFunction",
        "cols": ", ".join([f"value{i}" for i in range(90)]),
        "query_sql": "SELECT YearFunction({cols}) FROM postgres_data.year_predict_origin_test LIMIT {n};",
    },
    "sst-2": {
        "category": "NLP",
        "func_name": "SST2Function",
        "cols": "comment",
        "query_sql": "SELECT SST2Function(comment) FROM postgres_data.nlp_test LIMIT {n};",
    },
    "imdb": {
        "category": "NLP",
        "func_name": "IMDBFunction",
        "cols": "comment",
        "query_sql": "SELECT IMDBFunction(comment) FROM postgres_data.imdb_test LIMIT {n};",
    },
    "financial": {
        "category": "NLP",
        "func_name": "FinanceFunction",
        "cols": "comment",
        "query_sql": "SELECT FinanceFunction(comment) FROM postgres_data.financial_phrasebank_test LIMIT {n};",
    },
    "imagenet": {
        "category": "Image",
        "func_name": "ImagenetFunction",
        "cols": "image_path",
        "query_sql": "SELECT ImagenetFunction(image_path) FROM postgres_data.imagenet_image_table LIMIT {n};",
    },
    "cifar-10": {
        "category": "Image",
        "func_name": "CifarFunction",
        "cols": "image_path",
        "query_sql": "SELECT CifarFunction(image_path) FROM postgres_data.cifar_image_table LIMIT {n};",
    },
    "stanford_dogs": {
        "category": "Image",
        "func_name": "DogsFunction",
        "cols": "image_path",
        "query_sql": "SELECT DogsFunction(image_path) FROM postgres_data.stanford_dogs_image_table LIMIT {n};",
    },
}


def setup_evadb_function(ds_name, ds_config):
    """Create evadb function for a dataset (called once in main process)."""
    func_name = ds_config["func_name"]

    # Check if function already exists
    try:
        conn = evadb.connect()
        cursor = conn.cursor()

        # Ensure postgres_data database exists
        try:
            cursor.query("DROP DATABASE IF EXISTS postgres_data;").df()
        except:
            pass

        cursor.query(f"""
            CREATE DATABASE IF NOT EXISTS postgres_data
            WITH ENGINE = 'postgres',
            PARAMETERS = {json.dumps(DB_CONFIG)}
        """).df()

        # Try to drop existing function
        try:
            cursor.query(f"DROP FUNCTION IF EXISTS {func_name};").df()
        except:
            pass

        # Create function
        if "value" in ds_config["cols"]:
            n_cols = 384 if "slice" in ds_name else 90
            col_spec = ", ".join([f"value{i} NDARRAY FLOAT32" for i in range(n_cols)])
        elif "comment" in ds_config["cols"]:
            col_spec = "comment TEXT"
        else:
            col_spec = "image_path TEXT"

        # Get impl file path
        impl_files = {
            "SliceFunction": "morphingdb_test/series_test/slice_test/evadb_slice.py",
            "YearFunction": "morphingdb_test/series_test/year_predict_test/evadb_year_predict.py",
            "SST2Function": "morphingdb_test/text_test/sst2/evadb_sst2.py",
            "IMDBFunction": "morphingdb_test/text_test/imdb/evadb_imdb.py",
            "FinanceFunction": "morphingdb_test/text_test/financial_phrasebank/evadb_financial.py",
            "ImagenetFunction": "morphingdb_test/image_test/imagenet/evadb_resnet18.py",
            "CifarFunction": "morphingdb_test/image_test/cifar10/evadb_cifar.py",
            "DogsFunction": "morphingdb_test/image_test/stanford_dogs/evadb_alexnet_stanford_dog.py",
        }
        impl_file = impl_files.get(func_name, "")

        func_sql = f"""
            CREATE FUNCTION IF NOT EXISTS {func_name}
            INPUT  ({col_spec})
            OUTPUT (labels NDARRAY FLOAT32(1))
            TYPE Classification
            IMPL  '{impl_file}';
        """

        cursor.query(func_sql).df()
        print(f"  Function {func_name} created successfully")
        return True

    except Exception as e:
        print(f"  Function creation error: {str(e)[:150]}")
        return False


# =============================================================================
# Worker: runs one evadb query
# =============================================================================
def run_evadb_worker(task_id, ds_name, ds_config, row_count):
    """Execute one evadb query in a subprocess. Function already exists in catalog."""
    try:
        os.setpgid(0, 0)
    except OSError:
        pass

    try:
        sql = ds_config["query_sql"].format(cols=ds_config["cols"], n=row_count)

        t0 = time.perf_counter()
        conn = evadb.connect()
        result = conn.cursor().query(sql).df()
        elapsed = time.perf_counter() - t0

        print(f"  [OK] Task {task_id}: {elapsed:.2f}s")
        return {"task_id": task_id, "status": "success", "latency": elapsed}

    except Exception as e:
        err_msg = str(e)
        if len(err_msg) > 200:
            err_msg = err_msg[:200] + "..."
        print(f"  [FAIL] Task {task_id}: {err_msg}")
        return {"task_id": task_id, "status": "failed", "error": err_msg}


# =============================================================================
# Concurrency test with circuit breaker
# =============================================================================
def test_concurrency_level(concurrency_level, dataset_name, row_count):
    """Test a specific concurrency level with circuit breaker."""
    print(f"  Testing concurrency level: {concurrency_level}")

    ds_config = ALL_DATASETS[dataset_name]
    start_time = time.time()
    ctx = multiprocessing.get_context('spawn')
    pool = ctx.Pool(
        processes=concurrency_level,
        initializer=os.setpgid,
        initargs=(0, 0)
    )

    results = []
    try:
        async_results = []
        for i in range(concurrency_level):
            ar = pool.apply_async(run_evadb_worker, args=(i, dataset_name, ds_config, row_count))
            async_results.append(ar)

        for i, ar in enumerate(async_results):
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
    parser = argparse.ArgumentParser(description="EvaDB Throughput Test")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Test a single dataset")
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
        datasets_to_test = ALL_DATASETS

    print("=" * 80)
    print("EvaDB Throughput Test (Max Concurrency)")
    print(f"Upper limit: {MAX_CONCURRENCY_LIMIT} threads")
    print(f"Circuit breaker: {CIRCUIT_BREAKER_SEC}s (5 minutes)")
    print(f"Row count: {args.row_count}")
    print(f"Datasets: {', '.join(datasets_to_test.keys())}")
    print("=" * 80)

    all_results = {}

    for ds_name, ds_info in datasets_to_test.items():
        print(f"\n{'=' * 60}")
        print(f"  Dataset: {ds_name} ({ds_info['category']})")
        print(f"{'=' * 60}")

        # Create function once in main process
        print("\n  Setting up evadb function...")
        if not setup_evadb_function(ds_name, ds_info):
            print(f"  Skipping {ds_name} - function setup failed")
            continue

        # Warmup
        print("  Running warmup query...")
        try:
            warmup_sql = ds_info["query_sql"].format(cols=ds_info["cols"], n=10)
            evadb.connect().cursor().query(warmup_sql).df()
            print("  Warmup completed")
        except Exception as e:
            print(f"  Warmup failed: {str(e)[:100]}")
            print(f"  Skipping {ds_name} - warmup failed")
            continue

        time.sleep(2)

        max_conc = find_max_concurrency(ds_name, args.row_count)

        all_results[ds_name] = {
            "category": ds_info["category"],
            "row_count": args.row_count,
            "max_concurrency": max_conc,
        }

        print(f"\n  RESULT: {ds_name} MAX_CONCURRENCY = {max_conc}")

    # Summary
    print(f"\n{'=' * 80}")
    print("  SUMMARY: EvaDB Throughput (Max Concurrency)")
    print(f"{'=' * 80}")
    print(f"  {'Dataset':<20} {'Category':<10} {'Max Concurrency':<20}")
    print("  " + "-" * 50)
    for ds_name, res in all_results.items():
        print(f"  {ds_name:<20} {res['category']:<10} {res['max_concurrency']:<20}")
    print(f"{'=' * 80}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(ensure_dir(THROUGHPUT_RESULTS_DIR), f"evadb_throughput_results_{timestamp}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "test": "evadb_throughput",
            "max_limit": MAX_CONCURRENCY_LIMIT,
            "circuit_breaker_sec": CIRCUIT_BREAKER_SEC,
            "row_count": args.row_count,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
