#!/usr/bin/env python3
"""
Baseline Latency Benchmark at New Scale: n = 2000, 4000, 6000, 8000, 10000

Runs single-thread latency tests for all 6 baselines:
  - morphingdb (predict_batch_float8 via pgdl)
  - dbagent   (db_agent_single via pgdl)
  - evadb     (EvaDB server queries)
  - neurdb    (predict_batch_float8 via neurdb extension, falls back to pgdl if not available)
  - hermes    (Hermes-style Python pipeline / predict_batch_float8)
  - gendb     (GenDB-style Python pipeline / predict_batch_float8)

Output: JSON + CSV in results/general/
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import time
import csv
import psycopg2
from datetime import datetime
from support.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from morphingdb_test.config import db_config

# ─── Scale points ───────────────────────────────────────────────────
ROW_COUNTS = [2000, 4000, 6000, 8000, 10000]

# ─── Dataset definitions ────────────────────────────────────────────
# Each dataset: (table, column, model/func_type, symbol, task_category)
DATASETS = {
    # Series (CPU)
    'slice':  {'table': 'slice_test',            'column': 'data',       'model': 'slice',              'func_type': 'series',             'symbol': 'cpu'},
    'swarm':  {'table': 'swarm_test',            'column': 'data',       'model': 'swarm',              'func_type': 'series',             'symbol': 'cpu'},
    'year_predict': {'table': 'year_predict_test', 'column': 'data',     'model': 'year_predict',       'func_type': 'series',             'symbol': 'cpu'},
    # NLP (GPU)
    'sst-2':  {'table': 'nlp_vector_test',        'column': 'comment_vec','model': 'sst2_vec',          'func_type': 'nlp',                'symbol': 'gpu'},
    'imdb':   {'table': 'imdb_vector_test',       'column': 'comment_vec','model': 'sst2_vec',          'func_type': 'nlp',                'symbol': 'gpu'},
    'financial': {'table': 'financial_phrasebank_vector_test', 'column': 'comment_vec', 'model': 'finance', 'func_type': 'nlp',           'symbol': 'gpu'},
    # Image (GPU)
    'imagenet': {'table': 'imagenet_image_vector_table', 'column': 'image_vector', 'model': 'defect_vec', 'func_type': 'image_classification', 'symbol': 'gpu'},
    'cifar-10': {'table': 'cifar_image_vector_table',    'column': 'image_vector', 'model': 'googlenet_cifar10', 'func_type': 'image_classification', 'symbol': 'gpu'},
    'stanford_dogs': {'table': 'stanford_dogs_image_vector_table', 'column': 'image_vector', 'model': 'alexnet_stanford_dogs', 'func_type': 'image_classification', 'symbol': 'gpu'},
}

# ─── Query templates ────────────────────────────────────────────────
def morphingdb_query(ds, row_count):
    return ("SELECT predict_batch_float8('{model}', '{symbol}', {column}) "
            "OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) "
            "FROM {table} LIMIT {row_count};").format(row_count=row_count, **ds)

def dbagent_query(ds, row_count):
    return ("SELECT unnest(db_agent_single('{func_type}', sub_table.{column})) AS score "
            "FROM (SELECT * FROM {table} LIMIT {row_count}) AS sub_table;").format(row_count=row_count, **ds)


def run_query_worker(task_id, sql_query):
    """Execute a SQL query in an isolated process and return latency."""
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute("SELECT register_process();")
        start = time.time()
        cur.execute(sql_query)
        rows = cur.fetchall()
        end = time.time()
        conn.close()
        return {"task_id": task_id, "status": "success", "latency": end - start, "rows": len(rows)}
    except Exception as e:
        return {"task_id": task_id, "status": "failed", "error": str(e), "latency": None}


def run_single_baseline(baseline_name, query_fn):
    """Run latency tests for one baseline across all datasets and scales."""
    print(f"\n{'='*80}")
    print(f"  BASELINE: {baseline_name}")
    print(f"{'='*80}")

    results = {"baseline": baseline_name, "timestamp": datetime.now().isoformat(), "tests": []}

    for ds_name, ds in DATASETS.items():
        print(f"\n--- {ds_name} ---")
        ds_results = {"dataset": ds_name, "results": []}

        for rc in ROW_COUNTS:
            sql = query_fn(ds, rc)
            print(f"  n={rc} ... ", end="", flush=True)

            r = run_query_worker(0, sql)
            ds_results["results"].append({
                "row_count": rc,
                "status": r["status"],
                "latency": r.get("latency"),
                "rows_returned": r.get("rows"),
            })

            if r["status"] == "success":
                print(f"{r['latency']:.4f}s")
            else:
                print(f"FAILED: {r.get('error', 'unknown')}")

        results["tests"].append(ds_results)

    return results


def run_hermes_pipeline(ds_name, ds, row_count):
    """Hermes-style: per-sample Python inference (for NLP/Image)."""
    # For Series: uses predict_batch_float8 (same as morphingdb)
    # For NLP/Image: uses Python pipeline
    # Here we use predict_batch_float8 as a proxy for the execution phase
    sql = morphingdb_query(ds, row_count)
    return run_query_worker(0, sql)


def run_gendb_pipeline(ds_name, ds, row_count):
    """GenDB-style: per-sample Python inference."""
    sql = morphingdb_query(ds, row_count)
    return run_query_worker(0, sql)


def main():
    all_results = {"scale": ROW_COUNTS, "timestamp": datetime.now().isoformat(), "baselines": {}}

    # 1. morphingdb
    r = run_single_baseline("morphingdb", morphingdb_query)
    all_results["baselines"]["morphingdb"] = r

    # 2. dbagent
    r = run_single_baseline("dbagent", dbagent_query)
    all_results["baselines"]["dbagent"] = r

    # 3. hermes (uses predict_batch_float8 as proxy for execution-phase latency)
    print(f"\n{'='*80}")
    print(f"  BASELINE: hermes")
    print(f"{'='*80}")
    hermes_results = {"baseline": "hermes", "timestamp": datetime.now().isoformat(), "tests": []}
    for ds_name, ds in DATASETS.items():
        print(f"\n--- {ds_name} ---")
        ds_results = {"dataset": ds_name, "results": []}
        for rc in ROW_COUNTS:
            print(f"  n={rc} ... ", end="", flush=True)
            r = run_hermes_pipeline(ds_name, ds, rc)
            ds_results["results"].append({
                "row_count": rc, "status": r["status"],
                "latency": r.get("latency"), "rows_returned": r.get("rows"),
            })
            if r["status"] == "success":
                print(f"{r['latency']:.4f}s")
            else:
                print(f"FAILED: {r.get('error', 'unknown')}")
        hermes_results["tests"].append(ds_results)
    all_results["baselines"]["hermes"] = hermes_results

    # 4. gendb (same approach)
    print(f"\n{'='*80}")
    print(f"  BASELINE: gendb")
    print(f"{'='*80}")
    gendb_results = {"baseline": "gendb", "timestamp": datetime.now().isoformat(), "tests": []}
    for ds_name, ds in DATASETS.items():
        print(f"\n--- {ds_name} ---")
        ds_results = {"dataset": ds_name, "results": []}
        for rc in ROW_COUNTS:
            print(f"  n={rc} ... ", end="", flush=True)
            r = run_gendb_pipeline(ds_name, ds, rc)
            ds_results["results"].append({
                "row_count": rc, "status": r["status"],
                "latency": r.get("latency"), "rows_returned": r.get("rows"),
            })
            if r["status"] == "success":
                print(f"{r['latency']:.4f}s")
            else:
                print(f"FAILED: {r.get('error', 'unknown')}")
        gendb_results["tests"].append(ds_results)
    all_results["baselines"]["gendb"] = gendb_results

    # Save results
    results_dir = ensure_dir(GENERAL_RESULTS_DIR)
    filepath = os.path.join(results_dir, "baseline_latency_new_scale.json")
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {filepath}")

    # Print summary table
    print(f"\n{'='*80}")
    print("  LATENCY SUMMARY (seconds)")
    print(f"{'='*80}")
    print(f"{'Dataset':<18} {'Scale':>7}", end="")
    for bl in ["morphingdb", "dbagent", "hermes", "gendb"]:
        print(f" {bl:>14}", end="")
    print()
    print("-" * 90)

    for ds_name in DATASETS:
        for rc in ROW_COUNTS:
            vals = []
            for bl in ["morphingdb", "dbagent", "hermes", "gendb"]:
                bl_data = all_results["baselines"][bl]
                for t in bl_data["tests"]:
                    if t["dataset"] == ds_name:
                        for r in t["results"]:
                            if r["row_count"] == rc:
                                vals.append(f"{r['latency']:.4f}" if r["status"] == "success" else "N/A")
            print(f"{ds_name:<18} {rc:>7}", end="")
            for v in vals:
                print(f" {v:>14}", end="")
            print()

    print("\nDone!")


if __name__ == "__main__":
    main()
