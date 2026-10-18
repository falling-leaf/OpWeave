#!/usr/bin/env python3
import os
import sys
import time
import json
import multiprocessing
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import psycopg2
from morphingdb_test.config import db_config
from tools.test_result_paths import THROUGHPUT_RESULTS_DIR, ensure_dir

CONCURRENCY_LEVELS = [1, 5, 10]
N_ROWS = 20
COOL_DOWN = 15


def build_sql(n=N_ROWS):
    return (
        f"SELECT unnest(db_agent_single('musique', sub_table.reasoning_vec)) AS score "
        f"FROM (SELECT * FROM musique_vector_test ORDER BY id LIMIT {n}) AS sub_table;"
    )


def run_worker(task_id, sql_query):
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
        return {
            "task_id": task_id,
            "status": "failed",
            "latency": None,
            "rows": 0,
            "error": str(e),
        }
    finally:
        if conn is not None:
            try:
                conn.close()
            except:
                pass


def test_level(concurrency, sql_query):
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
            r = ar.get(timeout=3600)
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

    if success:
        latencies = [r["latency"] for r in success]
        avg_lat = sum(latencies) / len(latencies)
    else:
        avg_lat = None

    status_tag = f"{len(success)}/{concurrency} OK"
    if failed:
        status_tag += f", {len(failed)} FAILED"

    print(f"wall={wall:.1f}s  {status_tag}")
    for f in failed[:5]:
        print(f"      -> task {f['task_id']}: {str(f.get('error','?'))[:120]}")

    return {
        "concurrency": concurrency,
        "wall_time": round(wall, 4),
        "total_workers": concurrency,
        "success_count": len(success),
        "fail_count": len(failed),
        "avg_latency": round(avg_lat, 4) if avg_lat is not None else None,
        "results": results,
    }


def binary_search(low_pass, high_fail, sql_query):
    checked = []
    lo, hi = low_pass + 1, high_fail - 1
    best = low_pass
    while lo <= hi:
        mid = (lo + hi) // 2
        print(f"\n  --- Binary concurrency: {mid} ---")
        r = test_level(mid, sql_query)
        checked.append(r)
        if r["fail_count"] == 0:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
        if lo <= hi:
            print(f"    Cooling {COOL_DOWN}s...")
            time.sleep(COOL_DOWN)
    return best, checked


def main():
    sql = build_sql()

    print("=" * 70)
    print(f"  DBAgent Musique Throughput Profile — {datetime.now().isoformat()}")
    print(f"  Dataset: Musique (reasoning)")
    print(f"  SQL: db_agent_single('musique', reasoning_vec)")
    print(f"  n={N_ROWS} rows per worker")
    print(f"  Concurrency levels: {CONCURRENCY_LEVELS}")
    print(f"  Cutoff: none (DBAgent defines the reference wall time)")
    print("=" * 70)

    all_results = []
    pass_levels = []
    fail_levels = []
    for idx, level in enumerate(CONCURRENCY_LEVELS):
        print(f"\n  --- Concurrency: {level} ---")
        r = test_level(level, sql)
        all_results.append(r)
        if r["fail_count"] == 0:
            pass_levels.append(level)
        else:
            fail_levels.append(level)
        if idx < len(CONCURRENCY_LEVELS) - 1:
            print(f"    Cooling {COOL_DOWN}s...")
            time.sleep(COOL_DOWN)

    max_pass = max(pass_levels) if pass_levels else 0
    min_fail = min(fail_levels) if fail_levels else None
    binary_results = []
    if max_pass and min_fail and min_fail - max_pass > 1:
        best, binary_results = binary_search(max_pass, min_fail, sql)
        max_pass = best
        all_results.extend(binary_results)

    all_results = sorted(all_results, key=lambda x: x["concurrency"])
    max_success_wall = max((r["wall_time"] for r in all_results if r["fail_count"] == 0), default=None)

    out_dir = ensure_dir(THROUGHPUT_RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f"dbagent_musique_throughput_profile_{ts}.json")
    with open(path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "dataset": "musique_reasoning",
            "sql": sql,
            "n_rows": N_ROWS,
            "concurrency_levels": CONCURRENCY_LEVELS,
            "max_passing_concurrency": max_pass,
            "max_success_wall_time": max_success_wall,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {path}")
    print(f"Max passing concurrency: {max_pass}")
    print(f"Reference wall time for other systems: {max_success_wall}")


if __name__ == "__main__":
    main()
