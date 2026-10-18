#!/usr/bin/env python3
"""
EvaDB Command Execution Test for 9 Datasets

Tests evadb query execution latency at n=1000,2000,5000,10000
across all 9 base datasets. Results saved to JSON, NOT updated in paper tables.

Strategy:
- Series (slice): uses custom Python classifier (evadb_slice.py) - works
- Other Series (swarm, year_predict): need similar Python files
- NLP/Image: uses custom Python classifiers with local models

Usage:
    python test_evadb_throughput.py [--dataset <name>]
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import evadb

# =============================================================================
# Configuration
# =============================================================================
DB_CONFIG = {
    "database": "postgres",
    "host": "localhost",
    "port": "5432",
    "user": "why",
    "password": "123456"
}

TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from tools.test_result_paths import THROUGHPUT_RESULTS_DIR, ensure_dir

RESULT_DIR = ensure_dir(THROUGHPUT_RESULTS_DIR)
os.makedirs(RESULT_DIR, exist_ok=True)

N_LIST = [1000, 2000, 5000, 10000]

# =============================================================================
# Dataset configurations
# =============================================================================

# Helper to build column spec for series datasets
def build_cols(n_cols):
    return ", ".join([f"value{i} NDARRAY FLOAT32" for i in range(n_cols)])

def build_query_cols(n_cols):
    return ", ".join([f"value{i}" for i in range(n_cols)])

DATASETS = {
    # --- SERIES (CPU) ---
    "slice": {
        "category": "Series",
        "hardware": "cpu",
        "n_cols": 384,
        "table": "slice_origin_test",
        "func_name": "SliceFunction",
        "impl_file": "morphingdb_test/series_test/slice_test/evadb_slice.py",
        "query_sql": "SELECT {func}({cols}) FROM postgres_data.{table} LIMIT {n};",
    },
    "swarm": {
        "category": "Series",
        "hardware": "cpu",
        "n_cols": None,  # uses array column
        "table": "swarm_origin_test",
        "func_name": "SwarmFunction",
        "impl_file": "morphingdb_test/series_test/swarm_test/evadb_swarm.py",
        "query_sql": "SELECT {func}(data) FROM postgres_data.{table} LIMIT {n};",
        "single_col": "data",
    },
    "year_predict": {
        "category": "Series",
        "hardware": "cpu",
        "n_cols": 90,
        "table": "year_predict_origin_test",
        "func_name": "YearFunction",
        "impl_file": "morphingdb_test/series_test/year_predict_test/evadb_year_predict.py",
        "query_sql": "SELECT {func}({cols}) FROM postgres_data.{table} LIMIT {n};",
    },
    # --- NLP (GPU) ---
    "sst-2": {
        "category": "NLP",
        "hardware": "gpu",
        "table": "nlp_test",
        "func_name": "SST2Function",
        "impl_file": "morphingdb_test/text_test/sst2/evadb_sst2.py",
        "query_sql": "SELECT {func}(comment) FROM postgres_data.{table} LIMIT {n};",
    },
    "imdb": {
        "category": "NLP",
        "hardware": "gpu",
        "table": "imdb_test",
        "func_name": "IMDBFunction",
        "impl_file": "morphingdb_test/text_test/imdb/evadb_imdb.py",
        "query_sql": "SELECT {func}(comment) FROM postgres_data.{table} LIMIT {n};",
    },
    "financial": {
        "category": "NLP",
        "hardware": "gpu",
        "table": "financial_phrasebank_test",
        "func_name": "FinanceFunction",
        "impl_file": "morphingdb_test/text_test/financial_phrasebank/evadb_financial.py",
        "query_sql": "SELECT {func}(comment) FROM postgres_data.{table} LIMIT {n};",
    },
    # --- IMAGE (GPU) ---
    "imagenet": {
        "category": "Image",
        "hardware": "gpu",
        "table": "imagenet_image_table",
        "func_name": "ImagenetFunction",
        "impl_file": "morphingdb_test/image_test/imagenet/evadb_resnet18.py",
        "query_sql": "SELECT {func}(image_path) FROM postgres_data.{table} LIMIT {n};",
        "single_col": "image_path",
    },
    "cifar-10": {
        "category": "Image",
        "hardware": "gpu",
        "table": "cifar_image_table",
        "func_name": "CifarFunction",
        "impl_file": "morphingdb_test/image_test/cifar10/evadb_cifar.py",
        "query_sql": "SELECT {func}(image_path) FROM postgres_data.{table} LIMIT {n};",
        "single_col": "image_path",
    },
    "stanford_dogs": {
        "category": "Image",
        "hardware": "gpu",
        "table": "stanford_dogs_image_table",
        "func_name": "DogsFunction",
        "impl_file": "morphingdb_test/image_test/stanford_dogs/evadb_alexnet_stanford_dog.py",
        "query_sql": "SELECT {func}(image_path) FROM postgres_data.{table} LIMIT {n};",
        "single_col": "image_path",
    },
}


def setup_evadb():
    """Create evadb connection and connect to PostgreSQL."""
    conn = evadb.connect()
    cursor = conn.cursor()

    try:
        cursor.query("DROP DATABASE IF EXISTS postgres_data;").df()
    except:
        pass

    cursor.query(f"""
        CREATE DATABASE IF NOT EXISTS postgres_data
        WITH ENGINE = 'postgres',
        PARAMETERS = {json.dumps(DB_CONFIG)}
    """).df()
    return conn, cursor


def create_function(cursor, ds_name, ds_config):
    """Create evadb function for a dataset."""
    func_name = ds_config["func_name"]
    impl_file = ds_config["impl_file"]
    n_cols = ds_config.get("n_cols")

    if n_cols:
        col_spec = build_cols(n_cols)
        out_spec = f"OUTPUT (labels NDARRAY FLOAT32(1))"
    else:
        # For text/image, determine input type from table
        if "image" in ds_config["table"]:
            col_spec = "data NDARRAY UINT8"
            out_spec = "OUTPUT (labels NDARRAY FLOAT32(1000))"
        else:
            col_spec = "comment TEXT"
            out_spec = "OUTPUT (labels NDARRAY FLOAT32(3))"

    func_sql = f"""
        CREATE FUNCTION IF NOT EXISTS {func_name}
        INPUT  ({col_spec})
        {out_spec}
        TYPE Classification
        IMPL  '{impl_file}';
    """

    try:
        cursor.query(func_sql).df()
        return True
    except Exception as e:
        print(f"    Function creation failed: {str(e)[:100]}")
        return False


def test_dataset(ds_name, ds_config, n_list):
    """Test a single dataset at all n values."""
    print(f"\n{'='*60}")
    print(f"  Dataset: {ds_name} ({ds_config['category']})")
    print(f"  Hardware: {ds_config['hardware']}")
    print(f"{'='*60}")

    # Set hardware
    if ds_config["hardware"] == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Connect
    print("\n  Setting up evadb...")
    conn, cursor = setup_evadb()

    # Create function
    print("  Creating evadb function...")
    if not create_function(cursor, ds_name, ds_config):
        return None

    # Warmup
    print("  Running warmup...")
    n_cols = ds_config.get("n_cols")
    single_col = ds_config.get("single_col")
    if single_col:
        cols = single_col
    elif n_cols:
        cols = build_query_cols(n_cols)
    else:
        # Determine column name based on table structure
        if "image" in ds_config["table"]:
            cols = "image_path"
        else:
            cols = "comment"
    warmup_sql = ds_config["query_sql"].format(
        func=ds_config["func_name"], cols=cols, table=ds_config["table"], n=10)
    try:
        cursor.query(warmup_sql).df()
        print("  Warmup OK")
    except Exception as e:
        print(f"  Warmup failed: {str(e)[:100]}")

    # Test at each n
    results = {}
    for n in n_list:
        sql = ds_config["query_sql"].format(
            func=ds_config["func_name"], cols=cols, table=ds_config["table"], n=n)
        start_time = time.time()
        try:
            result = cursor.query(sql).df()
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"    n={n}: {elapsed:.2f}s (rows={len(result)})")
            results[n] = {"latency": elapsed, "success": True}
        except Exception as e:
            end_time = time.time()
            elapsed = end_time - start_time
            err_msg = str(e)[:80]
            print(f"    n={n}: FAILED after {elapsed:.2f}s - {err_msg}")
            results[n] = {"latency": elapsed, "success": False, "error": err_msg}
        time.sleep(2)

    return results


def main():
    parser = argparse.ArgumentParser(description="EvaDB Command Execution Test")
    parser.add_argument("--dataset", type=str, default=None, help="Test a single dataset")
    parser.add_argument("--n-list", type=str, default="1000,2000,5000,10000",
                        help="Comma-separated n values (default: 1000,2000,5000,10000)")
    args = parser.parse_args()

    n_list = [int(x.strip()) for x in args.n_list.split(",")]

    if args.dataset:
        if args.dataset not in DATASETS:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {', '.join(DATASETS.keys())}")
            sys.exit(1)
        datasets_to_test = {args.dataset: DATASETS[args.dataset]}
    else:
        datasets_to_test = DATASETS

    print("=" * 80)
    print("EvaDB Command Execution Test")
    print(f"n values: {n_list}")
    print(f"Datasets: {', '.join(datasets_to_test.keys())}")
    print("=" * 80)

    all_results = {}

    for ds_name, ds_info in datasets_to_test.items():
        try:
            results = test_dataset(ds_name, ds_info, n_list)
            if results:
                all_results[ds_name] = {
                    "category": ds_info["category"],
                    "hardware": ds_info["hardware"],
                    "results": results,
                }
        except Exception as e:
            print(f"\n  ERROR testing {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*80}")
    print("  SUMMARY: EvaDB Command Execution Test")
    print(f"{'='*80}")
    header = f"  {'Dataset':<15} {'Category':<10}"
    for n in n_list:
        header += f" {'n='+str(n):<12}"
    print(header)
    print("  " + "-" * (25 + len(n_list)*12))
    for ds_name, res in all_results.items():
        row = f"  {ds_name:<15} {res['category']:<10}"
        for n in n_list:
            r = res["results"].get(n, {})
            if r.get("success"):
                row += f" {r['latency']:.2f}s{'':<7}"
            else:
                row += f" {'FAIL':<12}"
        print(row)
    print(f"{'='*80}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(RESULT_DIR, f"evadb_throughput_results_{timestamp}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "test": "evadb_command_execution_test",
            "n_values": n_list,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
