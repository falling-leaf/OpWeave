#!/usr/bin/env python3
"""
DBAgent Throughput Profile Test
================================
Test DBAgent at fixed concurrency levels: 1, 5, 10, 15, 20.
Record actual execution time for each level (no pass/fail judgment).
Report predict_error and other failures as-is.

Datasets: slice, swarm, year_predict (CPU) + sst-2, imdb, financial (GPU NLP)
          + imagenet, cifar-10, stanford_dogs (GPU Image)
n=1000 per worker. Results saved to test/results/.
"""

import sys, os, time, json, multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import psycopg2
from morphingdb_test.config import db_config
from support.test_result_paths import THROUGHPUT_RESULTS_DIR, ensure_dir

# ─── Configuration ──────────────────────────────────────────────────
CONCURRENCY_LEVELS = [1, 5, 10, 15, 20]
N_ROWS = 1000
COOL_DOWN = 10  # seconds between concurrency levels

# Dataset configs
DATASETS = [
    # Series (CPU)
    {"name": "slice",       "category": "Series", "hw": "cpu",
     "model": "slice",              "table": "slice_test",
     "column": "data",        "func_type": "series"},
    {"name": "swarm",       "category": "Series", "hw": "cpu",
     "model": "swarm",              "table": "swarm_test",
     "column": "data",        "func_type": "series"},
    {"name": "year_predict","category": "Series", "hw": "cpu",
     "model": "year_predict",       "table": "year_predict_test",
     "column": "data",        "func_type": "series"},
    # NLP (GPU)
    {"name": "sst-2",       "category": "NLP",    "hw": "gpu",
     "model": "sst2_vec",           "table": "nlp_vector_test",
     "column": "comment_vec", "func_type": "nlp"},
    {"name": "imdb",        "category": "NLP",    "hw": "gpu",
     "model": "sst2_vec",           "table": "imdb_vector_test",
     "column": "comment_vec", "func_type": "nlp"},
    {"name": "financial",   "category": "NLP",    "hw": "gpu",
     "model": "finance",            "table": "financial_phrasebank_vector_test",
     "column": "comment_vec", "func_type": "nlp"},
    # Image (GPU)
    {"name": "imagenet",    "category": "Image",  "hw": "gpu",
     "model": "defect_vec",         "table": "imagenet_image_vector_table",
     "column": "image_vector", "func_type": "image_classification"},
    {"name": "cifar-10",    "category": "Image",  "hw": "gpu",
     "model": "googlenet_cifar10",  "table": "cifar_image_vector_table",
     "column": "image_vector", "func_type": "image_classification"},
    {"name": "stanford_dogs","category": "Image", "hw": "gpu",
     "model": "alexnet_stanford_dogs","table": "stanford_dogs_image_vector_table",
     "column": "image_vector", "func_type": "image_classification"},
]


def build_sql_predict_batch(ds):
    """Build predict_batch_float8 SQL for DBAgent throughput test."""
    return (
        f"SELECT predict_batch_float8('{ds['model']}', '{ds['hw']}', {ds['column']}) "
        f"OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) "
        f"FROM {ds['table']} LIMIT {N_ROWS};"
    )


def build_sql_dbagent_single(ds):
    """Build db_agent_single SQL for DBAgent throughput test."""
    return (
        f"SELECT unnest(db_agent_single('{ds['func_type']}', sub_table.{ds['column']})) AS score "
        f"FROM (SELECT * FROM {ds['table']} LIMIT {N_ROWS}) AS sub_table;"
    )


def run_worker(task_id, sql_query):
    """Single worker: connect, register, execute SQL, return timing + status."""
    try:
        os.setpgid(0, 0)
    except OSError:
        pass

    conn = None
    try:
        conn = psycopg2.connect(**db_config, connect_timeout=10)
        cur = conn.cursor()
        cur.execute("SELECT register_process();")

        t0 = time.perf_counter()
        cur.execute(sql_query)
        rows = cur.fetchall()
        elapsed = time.perf_counter() - t0

        conn.close()
        conn = None

        return {
            "task_id": task_id,
            "status": "success",
            "latency": round(elapsed, 4),
            "rows": len(rows),
            "error": None,
        }
    except Exception as e:
        err_str = str(e)
        return {
            "task_id": task_id,
            "status": "failed",
            "latency": None,
            "rows": 0,
            "error": err_str,
        }
    finally:
        if conn is not None:
            try:
                conn.close()
            except:
                pass


def test_level(concurrency, sql_query):
    """
    Test a specific concurrency level.
    Returns wall time, per-worker results, and summary stats.
    NO pass/fail judgment — just raw data.
    """
    print(f"    Testing {concurrency:>3d} concurrent workers...", end=" ", flush=True)

    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=concurrency, initializer=os.setpgid, initargs=(0, 0))

    results = []
    async_results = []
    for i in range(concurrency):
        ar = pool.apply_async(run_worker, args=(i, sql_query))
        async_results.append(ar)

    for i, ar in enumerate(async_results):
        try:
            r = ar.get(timeout=3600)  # 1 hour per worker — effectively no timeout
            results.append(r)
        except Exception as e:
            results.append({
                "task_id": i,
                "status": "failed",
                "latency": None,
                "rows": 0,
                "error": str(e)[:200],
            })

    pool.terminate()
    pool.join()

    wall = time.time() - t0

    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    # Stats from successful workers only
    if success:
        latencies = [r["latency"] for r in success]
        avg_lat = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)
    else:
        avg_lat = min_lat = max_lat = None

    status_tag = f"{len(success)}/{concurrency} OK"
    if failed:
        status_tag += f", {len(failed)} FAILED"

    print(f"wall={wall:.1f}s  {status_tag}")

    # Print errors inline
    for f in failed:
        err = f.get("error", "?")
        # Truncate long errors but keep key info
        if "predict error" in err.lower():
            err_display = f"PREDICT_ERROR (task {f['task_id']})"
        else:
            err_display = f"task {f['task_id']}: {err[:100]}"
        print(f"      -> {err_display}")

    return {
        "concurrency": concurrency,
        "wall_time": round(wall, 4),
        "total_workers": concurrency,
        "success_count": len(success),
        "fail_count": len(failed),
        "avg_latency": round(avg_lat, 4) if avg_lat is not None else None,
        "min_latency": round(min_lat, 4) if min_lat is not None else None,
        "max_latency": round(max_lat, 4) if max_lat is not None else None,
        "results": results,
    }


def test_dataset(ds, sql_builder):
    """Test all concurrency levels for one dataset."""
    ds_name = ds["name"]
    sql = sql_builder(ds)
    print(f"\n  Dataset: {ds_name} ({ds['category']}, {ds['hw']})")
    print(f"  Model: {ds['model']}")
    print(f"  SQL: {sql[:120]}...")

    level_results = []
    for level in CONCURRENCY_LEVELS:
        r = test_level(level, sql)
        level_results.append(r)

        if level < CONCURRENCY_LEVELS[-1]:
            print(f"    Cooling {COOL_DOWN}s...")
            time.sleep(COOL_DOWN)

    return {
        "dataset": ds_name,
        "category": ds["category"],
        "hardware": ds["hw"],
        "model": ds["model"],
        "sql": sql,
        "levels": level_results,
    }


def print_summary_table(all_results, sql_type):
    """Print a clean summary table."""
    print(f"\n{'='*90}")
    print(f"  DBAgent Throughput Profile — {sql_type}")
    print(f"  {'='*90}")

    # Header
    header = f"  {'Dataset':<14} {'Category':<10} {'HW':<4}"
    for level in CONCURRENCY_LEVELS:
        header += f" | {level:>8d} workers"
    print(header)
    print("  " + "-" * (14 + 10 + 4 + 12 * len(CONCURRENCY_LEVELS)))

    for ds_result in all_results:
        row = f"  {ds_result['dataset']:<14} {ds_result['category']:<10} {ds_result['hardware']:<4}"
        for lr in ds_result["levels"]:
            wall = lr["wall_time"]
            avg = lr["avg_latency"]
            ok = lr["success_count"]
            total = lr["total_workers"]
            if avg is not None:
                row += f" | {wall:>5.1f}s/{avg:>5.1f}s"
            else:
                row += f" | {wall:>5.1f}s/N/A"
            if ok < total:
                row += "!"
        print(row)

    print(f"  {'='*90}")
    print(f"  Format: wall_time/avg_latency per worker  (! = some workers failed)")
    print(f"  {'='*90}\n")

    # Error summary
    has_errors = False
    for ds_result in all_results:
        for lr in ds_result["levels"]:
            for r in lr["results"]:
                if r["status"] == "failed":
                    if not has_errors:
                        print(f"  Error details:")
                        has_errors = True
                    err = r.get("error", "unknown")
                    if "predict error" in err.lower():
                        print(f"    [{ds_result['dataset']}] concurrency={lr['concurrency']}, "
                              f"task {r['task_id']}: PREDICT_ERROR")
                    else:
                        print(f"    [{ds_result['dataset']}] concurrency={lr['concurrency']}, "
                              f"task {r['task_id']}: {err[:120]}")

    if has_errors:
        print()


def main():
    parser = argparse.ArgumentParser(description="DBAgent Throughput Profile Test")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Test a single dataset (e.g., 'slice')")
    parser.add_argument("--sql-type", type=str, default="predict_batch",
                        choices=["predict_batch", "db_agent_single"],
                        help="SQL function to use (default: predict_batch)")
    args = parser.parse_args()

    datasets = [d for d in DATASETS if d["name"] == args.dataset] if args.dataset else DATASETS
    sql_builder = (build_sql_predict_batch if args.sql_type == "predict_batch"
                   else build_sql_dbagent_single)

    print("=" * 70)
    print(f"  DBAgent Throughput Profile — {datetime.now().isoformat()}")
    print(f"  Datasets: {[d['name'] for d in datasets]}")
    print(f"  SQL type: {args.sql_type}")
    print(f"  Concurrency levels: {CONCURRENCY_LEVELS}")
    print(f"  n={N_ROWS} rows per worker")
    print("=" * 70)

    all_results = []
    for ds in datasets:
        result = test_dataset(ds, sql_builder)
        all_results.append(result)

    # Summary table
    print_summary_table(all_results, args.sql_type)

    # Save results
    out_dir = ensure_dir(THROUGHPUT_RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f"dbagent_throughput_profile_{args.sql_type}_{ts}.json")
    with open(path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "sql_type": args.sql_type,
            "concurrency_levels": CONCURRENCY_LEVELS,
            "n_rows": N_ROWS,
            "datasets": all_results,
        }, f, indent=2, default=str)
    print(f"Results saved to: {path}")


if __name__ == "__main__":
    import argparse
    main()
