#!/usr/bin/env python3
"""
DBAgent HotpotQA (Reasoning) Throughput Profile Test
=====================================================
Test DBAgent at fixed concurrency levels: 1, 5, 10, 15, 20.
Record actual execution time for each level (no pass/fail judgment).
Report predict_error and other failures as-is.

Dataset: HotpotQA (reasoning) — n=20 samples, db_agent_single('reasoning', ...).
Results saved to support/results/throughput/.
"""

import sys, os, time, json, multiprocessing, argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import psycopg2
from morphingdb_test.config import db_config
from support.test_result_paths import THROUGHPUT_RESULTS_DIR, ensure_dir

# ─── Configuration ──────────────────────────────────────────────────
CONCURRENCY_LEVELS = [1, 5, 10, 15, 20]
N_ROWS = 20  # reasoning_vector_test has only 20 rows
COOL_DOWN = 30  # seconds between concurrency levels (reasoning is heavy)


def build_sql(n=N_ROWS):
    """Build db_agent_single SQL for HotpotQA/reasoning."""
    return (
        f"SELECT unnest(db_agent_single('reasoning', sub_table.reasoning_vec)) AS score "
        f"FROM (SELECT * FROM reasoning_vector_test LIMIT {n}) AS sub_table;"
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


def main():
    sql = build_sql()

    print("=" * 70)
    print(f"  DBAgent HotpotQA Throughput Profile — {datetime.now().isoformat()}")
    print(f"  Dataset: HotpotQA (reasoning)")
    print(f"  SQL: db_agent_single('reasoning', reasoning_vec)")
    print(f"  n={N_ROWS} rows per worker")
    print(f"  Concurrency levels: {CONCURRENCY_LEVELS}")
    print("=" * 70)
    print(f"  SQL: {sql[:120]}...")

    all_results = []
    for level in CONCURRENCY_LEVELS:
        print(f"\n  --- Concurrency: {level} ---")
        r = test_level(level, sql)
        all_results.append(r)

        if level < CONCURRENCY_LEVELS[-1]:
            print(f"    Cooling {COOL_DOWN}s...")
            time.sleep(COOL_DOWN)

    # Summary table
    print(f"\n{'='*80}")
    print(f"  DBAgent HotpotQA Throughput Profile")
    print(f"  {'='*80}")

    header = f"  {'Concurrency':>12} | {'Wall Time':>10} | {'Avg Latency':>12} | {'Min':>8} | {'Max':>8} | {'OK':>6} | {'Failed':>6}"
    print(header)
    print("  " + "-" * (12 + 11 + 14 + 9 + 9 + 8 + 8))

    for lr in all_results:
        wall = lr["wall_time"]
        avg = lr["avg_latency"]
        mn = lr["min_latency"]
        mx = lr["max_latency"]
        ok = lr["success_count"]
        fail = lr["fail_count"]
        avg_str = f"{avg:.2f}s" if avg is not None else "N/A"
        mn_str = f"{mn:.2f}s" if mn is not None else "N/A"
        mx_str = f"{mx:.2f}s" if mx is not None else "N/A"
        print(f"  {lr['concurrency']:>12} | {wall:>9.2f}s | {avg_str:>12} | {mn_str:>8} | {mx_str:>8} | {ok:>6} | {fail:>6}")

    print(f"  {'='*80}")

    # Error details
    has_errors = False
    for lr in all_results:
        for r in lr["results"]:
            if r["status"] == "failed":
                if not has_errors:
                    print(f"\n  Error details:")
                    has_errors = True
                err = r.get("error", "unknown")
                if "predict error" in err.lower():
                    print(f"    concurrency={lr['concurrency']}, task {r['task_id']}: PREDICT_ERROR")
                elif "server closed" in err.lower():
                    print(f"    concurrency={lr['concurrency']}, task {r['task_id']}: SERVER_CLOSED (PG crash)")
                else:
                    print(f"    concurrency={lr['concurrency']}, task {r['task_id']}: {err[:120]}")

    if has_errors:
        print()

    # Save results
    out_dir = ensure_dir(THROUGHPUT_RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f"dbagent_hotpotqa_throughput_profile_{ts}.json")
    with open(path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "dataset": "hotpotqa_reasoning",
            "sql": sql,
            "n_rows": N_ROWS,
            "concurrency_levels": CONCURRENCY_LEVELS,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    main()
