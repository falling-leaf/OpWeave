#!/usr/bin/env python3
"""
DBAgent Latency Benchmark at New Scale: n = 2000, 4000, 6000, 8000, 10000
Runs db_agent_single queries only. Shorter timeout per query.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import json, time, psycopg2
from datetime import datetime
from tools.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir
from tools.general_extension import ensure_opt_off
from morphingdb_test.config import db_config

ROW_COUNTS = [2000, 4000, 6000, 8000, 10000]
# Paper main-result table uses n=1000, 2000, 5000, 10000 for image.
PAPER_IMAGE_ROW_COUNTS = [1000, 2000, 5000, 10000]
QUERY_TIMEOUT = 120  # per-query timeout in seconds

DATASETS = {
    'slice':  {'table': 'slice_test',            'column': 'data', 'func_type': 'series'},
    'swarm':  {'table': 'swarm_test',            'column': 'data', 'func_type': 'series'},
    'year_predict': {'table': 'year_predict_test', 'column': 'data', 'func_type': 'series'},
    'sst-2':  {'table': 'nlp_vector_test',        'column': 'comment_vec', 'func_type': 'nlp'},
    'imdb':   {'table': 'imdb_vector_test',       'column': 'comment_vec', 'func_type': 'nlp'},
    'financial': {'table': 'financial_phrasebank_vector_test', 'column': 'comment_vec', 'func_type': 'nlp'},
    'imagenet': {'table': 'imagenet_image_vector_table', 'column': 'image_vector', 'func_type': 'image_classification'},
    'cifar-10': {'table': 'cifar_image_vector_table', 'column': 'image_vector', 'func_type': 'image_classification'},
    'stanford_dogs': {'table': 'stanford_dogs_image_vector_table', 'column': 'image_vector', 'func_type': 'image_classification'},
}

def dbagent_query(ds, row_count):
    return ("SELECT unnest(db_agent_single('{func_type}', sub_table.{column})) AS score "
            "FROM (SELECT * FROM {table} LIMIT {row_count}) AS sub_table;").format(row_count=row_count, **ds)

def run_query(sql, timeout=QUERY_TIMEOUT, conn=None):
    """Execute SQL with timeout."""
    import signal
    class TimeoutError(Exception): pass
    def handler(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    owned_conn = conn is None
    try:
        if owned_conn:
            conn = psycopg2.connect(**db_config)
            cur = conn.cursor()
            cur.execute("SELECT register_process();")
        else:
            cur = conn.cursor()
        start = time.time()
        cur.execute(sql)
        rows = cur.fetchall()
        end = time.time()
        cur.close()
        if owned_conn:
            conn.close()
        signal.alarm(0)
        return {"status": "success", "latency": end - start, "rows": len(rows)}
    except TimeoutError:
        signal.alarm(0)
        if owned_conn and conn is not None:
            conn.close()
        return {"status": "timeout", "latency": None}
    except Exception as e:
        signal.alarm(0)
        if owned_conn and conn is not None:
            conn.close()
        return {"status": "failed", "error": str(e), "latency": None}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS.keys()))
    parser.add_argument("--paper-scale", action="store_true",
                        help="Use the image scales from the paper main-result table.")
    parser.add_argument("--reuse-connection", action="store_true",
                        help="Reuse a single PostgreSQL connection across all queries in this run.")
    args = parser.parse_args()
    ensure_opt_off()
    row_counts = PAPER_IMAGE_ROW_COUNTS if args.paper_scale else ROW_COUNTS

    selected = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS
    all_results = {
        "baseline": "dbagent",
        "scale": row_counts,
        "timestamp": datetime.now().isoformat(),
        "reuse_connection": args.reuse_connection,
        "tests": []
    }

    shared_conn = None
    if args.reuse_connection:
        shared_conn = psycopg2.connect(**db_config)
        cur = shared_conn.cursor()
        cur.execute("SELECT register_process();")
        cur.close()

    for ds_name, ds in selected.items():
        print(f"\n--- {ds_name} ---")
        ds_results = {"dataset": ds_name, "results": []}
        for rc in row_counts:
            sql = dbagent_query(ds, rc)
            print(f"  n={rc} ... ", end="", flush=True)
            r = run_query(sql, conn=shared_conn if args.reuse_connection else None)
            ds_results["results"].append({
                "row_count": rc, "status": r["status"],
                "latency": r.get("latency"), "rows_returned": r.get("rows"),
            })
            if r["status"] == "success":
                print(f"{r['latency']:.4f}s")
            else:
                print(f"{r['status'].upper()}")
        all_results["tests"].append(ds_results)

    if shared_conn is not None:
        shared_conn.close()

    results_dir = ensure_dir(GENERAL_RESULTS_DIR)
    suffix = "_reuse_conn" if args.reuse_connection else ""
    dataset_suffix = f"_{args.dataset}" if args.dataset else ""
    scale_suffix = "_paper" if args.paper_scale else ""
    filepath = os.path.join(results_dir, f"dbagent_latency_new_scale{dataset_suffix}{scale_suffix}{suffix}.json")
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {filepath}")

    # Summary
    print(f"\n{'Dataset':<18}", end="")
    for rc in row_counts:
        print(f" n={rc:<6}", end="")
    print()
    print("-" * 60)
    for t in all_results["tests"]:
        print(f"{t['dataset']:<18}", end="")
        for r in t["results"]:
            if r["status"] == "success":
                print(f" {r['latency']:<8.4f}", end="")
            else:
                print(f" {r['status']:<8}", end="")
        print()

if __name__ == "__main__":
    main()
