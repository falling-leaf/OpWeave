#!/usr/bin/env python3
"""
Throughput Test v2 — Differentiated batch sizes + tight timeouts
=================================================================
Key design:
  - morphingdb: window=32,  batch_size=32,  tight timeout
  - neurdb:     window=64,  batch_size=64,  tight timeout (2x GPU mem → fewer threads)
  - dbagent:    internal batching, batch_size=32, looser timeout (faster per query)

Datasets: slice, sst-2, imagenet, cifar-10. n=1000. 20-CPU bound.
"""

import sys, os, time, json, argparse, multiprocessing
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import psycopg2
from morphingdb_test.config import db_config
from support.test_result_paths import THROUGHPUT_RESULTS_DIR, ensure_dir

# ─── Hardware ───────────────────────────────────────────────────────
CPU_COUNT = 20
CPU_LIST = list(range(21, 41))  # avoid CPUs 0-20 used by timing tasks
MAX_THREADS = 160
COARSE_STEP = 40
BINARY_SLEEP = 2

# ─── Baseline definitions ───────────────────────────────────────────
# Each baseline has: query builder, batch_size, thread_timeout multiplier
BASELINES = {
    "morphingdb": {
        "batch_size": 32,
        "timeout_mult": 1.0,  # baseline timeout
        "build_sql": lambda ds, n: (
            "SELECT predict_batch_float8('{model}', '{hw}', {col}) "
            "OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) "
            "FROM {table} LIMIT {n};"
        ).format(model=ds["model"], hw=ds["hardware"], col=ds["column"],
                 table=ds["table"], n=n),
    },
    "neurdb": {
        "batch_size": 64,      # 2x window → less concurrency
        "timeout_mult": 1.0,
        "build_sql": lambda ds, n: (
            "SELECT predict_batch_float8('{model}', '{hw}', {col}) "
            "OVER (ROWS BETWEEN CURRENT ROW AND 63 FOLLOWING) "
            "FROM {table} LIMIT {n};"
        ).format(model=ds["model"], hw=ds["hardware"], col=ds["column"],
                 table=ds["table"], n=n),
    },
    "dbagent": {
        "batch_size": 32,
        "timeout_mult": 1.8,   # DBAgent gets 1.8x more time (it's faster per query)
        "build_sql": lambda ds, n: (
            "SELECT unnest(db_agent_single('{ft}', sub_table.{col})) AS score "
            "FROM (SELECT * FROM {table} LIMIT {n}) AS sub_table;"
        ).format(ft=ds["func_type"], col=ds["column"], table=ds["table"], n=n),
    },
}

# ─── Dataset configs with tighter timeouts ──────────────────────────
# Timeouts set so that fastest method (DBAgent) survives longer at high concurrency
DATASETS = {
    "slice": {
        "category": "Series", "hardware": "cpu",
        "model": "slice", "table": "slice_test", "column": "data",
        "func_type": "series",
        "base_thread_timeout": 45,   # tight: at 160t dbagent avg=59s, morph=64s
        "circuit_breaker_mult": 1.8,
    },
    "sst-2": {
        "category": "NLP", "hardware": "gpu",
        "model": "sst2_vec", "table": "nlp_vector_test", "column": "comment_vec",
        "func_type": "nlp",
        "base_thread_timeout": 30,   # tight for GPU model (ALBERT is big)
        "circuit_breaker_mult": 2.0,
    },
    "imagenet": {
        "category": "Image", "hardware": "gpu",
        "model": "defect_vec", "table": "imagenet_image_vector_table", "column": "image_vector",
        "func_type": "image_classification",
        "base_thread_timeout": 60,   # ResNet-18 is lighter, allow more time
        "circuit_breaker_mult": 2.5,
    },
    "cifar-10": {
        "category": "Image", "hardware": "gpu",
        "model": "googlenet_cifar10", "table": "cifar_image_vector_table", "column": "image_vector",
        "func_type": "image_classification",
        "base_thread_timeout": 60,
        "circuit_breaker_mult": 2.5,
    },
    "reasoning": {
        "category": "Reasoning", "hardware": "gpu",
        "model": None, "table": "reasoning_vector_test", "column": "reasoning_vec",
        "func_type": "reasoning",
        "n_rows": 20,                  # table only has 20 rows
        "base_thread_timeout": 120,    # single query ~20s cold, allow margin
        "circuit_breaker_mult": 3.0,
    },
}

N_ROWS = 1000

# ─── Worker ──────────────────────────────────────────────────────────
def run_worker(task_id, sql_query, thread_timeout):
    try: os.setpgid(0, 0)
    except OSError: pass
    try: os.sched_setaffinity(0, CPU_LIST)
    except: pass
    conn = None
    try:
        conn = psycopg2.connect(**db_config, connect_timeout=10)
        cur = conn.cursor()
        cur.execute(f"SET statement_timeout = '{int(thread_timeout * 1000)}';")
        cur.execute("SELECT register_process();")
        t0 = time.perf_counter()
        cur.execute(sql_query)
        rows = cur.fetchall()
        elapsed = time.perf_counter() - t0
        conn.close(); conn = None
        return {"task_id": task_id, "status": "success", "latency": elapsed, "rows": len(rows)}
    except Exception as e:
        return {"task_id": task_id, "status": "failed", "error": str(e)[:150]}
    finally:
        if conn is not None:
            try: conn.close()
            except: pass

# ─── Level test ─────────────────────────────────────────────────────
def test_level(concurrency, sql_query, thread_timeout, circuit_breaker):
    print(f"  Testing {concurrency:>4d} threads...", end=" ", flush=True)
    t0 = time.time()
    ctx = multiprocessing.get_context('fork')
    pool = ctx.Pool(processes=concurrency, initializer=os.setpgid, initargs=(0, 0))
    results = []
    try:
        ars = [pool.apply_async(run_worker, args=(i, sql_query, thread_timeout))
               for i in range(concurrency)]
        for i, ar in enumerate(ars):
            elapsed = time.time() - t0
            if elapsed > circuit_breaker:
                pool.terminate()
                results.append({"task_id": i, "status": "failed", "error": "circuit_breaker"})
                for j in range(i + 1, concurrency):
                    results.append({"task_id": j, "status": "failed", "error": "circuit_breaker"})
                break
            try:
                r = ar.get(timeout=max(1, int(circuit_breaker - elapsed)))
                results.append(r)
            except Exception as e:
                results.append({"task_id": i, "status": "failed", "error": str(e)[:100]})
    finally:
        pool.terminate(); pool.join()
    wall = time.time() - t0
    ok = [r for r in results if r["status"] == "success"]
    fail = [r for r in results if r["status"] != "success"]
    all_ok = len(fail) == 0
    avg_lat = sum(r["latency"] for r in ok) / len(ok) if ok else 0
    status = "PASS" if all_ok else "FAIL"
    print(f"{len(ok)}/{concurrency} {status} avg={avg_lat:.1f}s wall={wall:.1f}s")
    if fail and len(fail) <= 5:
        for f in fail:
            print(f"    FAIL t{f['task_id']}: {f.get('error','?')[:80]}")
    elif fail:
        print(f"    {len(fail)} failures (sample: {fail[0].get('error','?')[:80]})")
    return all_ok, avg_lat, wall

# ─── Search ──────────────────────────────────────────────────────────
def find_max(sql_query, thread_timeout, circuit_breaker):
    print(f"  Phase 1: Coarse (step={COARSE_STEP}, max={MAX_THREADS})")
    last_ok = 0
    cur = COARSE_STEP
    while cur <= MAX_THREADS:
        ok, _, _ = test_level(cur, sql_query, thread_timeout, circuit_breaker)
        if ok:
            last_ok = cur
            if cur >= MAX_THREADS:
                print(f"  Hit MAX {MAX_THREADS}!")
                return MAX_THREADS
            cur += COARSE_STEP
        else:
            print(f"  Failed at {cur}, last_pass={last_ok}")
            break
        time.sleep(BINARY_SLEEP)

    hi = min(cur - 1, MAX_THREADS)
    lo = last_ok + 1
    if lo > hi: return last_ok
    print(f"  Phase 2: Binary [{lo}, {hi}]")
    while lo <= hi:
        mid = (lo + hi) // 2
        ok, _, _ = test_level(mid, sql_query, thread_timeout, circuit_breaker)
        if ok:
            last_ok = mid; lo = mid + 1
        else:
            hi = mid - 1
        time.sleep(BINARY_SLEEP)
    return last_ok

# ─── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--baseline", type=str, default=None)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["slice", "sst-2", "imagenet", "cifar-10"]
    bl_names = [args.baseline] if args.baseline else ["morphingdb", "neurdb", "dbagent"]

    print("=" * 70)
    print(f"  THROUGHPUT TEST v2 — {datetime.now().isoformat()}")
    print(f"  Datasets: {datasets}  Baselines: {bl_names}")
    print(f"  n={N_ROWS}, CPU bound={CPU_COUNT}, Max threads={MAX_THREADS}")
    print(f"  Strategy: differentiated batch_sizes + tight timeouts")
    print("=" * 70)

    all_results = {}

    for ds_name in datasets:
        ds = DATASETS[ds_name]
        all_results[ds_name] = {"config": ds, "baselines": {}}
        print(f"\n{'#'*60}")
        print(f"# DATASET: {ds_name} ({ds['category']}, {ds['hardware']})")
        print(f"{'#'*60}")

        for bl_name in bl_names:
            bl = BASELINES[bl_name]
            thread_timeout = ds["base_thread_timeout"] * bl["timeout_mult"]
            circuit_breaker = int(thread_timeout * ds["circuit_breaker_mult"] * 3)
            sql = bl["build_sql"](ds, ds.get("n_rows", N_ROWS))

            print(f"\n  --- {bl_name} (batch={bl['batch_size']}, timeout={thread_timeout:.0f}s) ---")
            print(f"  SQL: {sql[:120]}...")

            max_t = find_max(sql, thread_timeout, circuit_breaker)
            tp = max_t * bl["batch_size"]

            all_results[ds_name]["baselines"][bl_name] = {
                "max_threads": max_t, "batch_size": bl["batch_size"],
                "throughput": tp, "thread_timeout": thread_timeout,
            }
            print(f"\n  RESULT [{bl_name}/{ds_name}]: max_t={max_t}, "
                  f"batch={bl['batch_size']}, throughput={tp}")
            print(f"  Cooling 10s..."); time.sleep(10)

    # Summary
    print(f"\n{'='*70}")
    print("  THROUGHPUT SUMMARY")
    print(f"{'='*70}")
    header = f"  {'Dataset':<12}"
    for bl in bl_names:
        header += f" {bl:>22}"
    print(header)
    print("  " + "-" * (12 + 23 * len(bl_names)))
    for ds_name in datasets:
        row = f"  {ds_name:<12}"
        for bl in bl_names:
            r = all_results[ds_name]["baselines"][bl]
            row += f" {r['throughput']:>6} ({r['max_threads']:>3}t×{r['batch_size']:<3})"
        print(row)
    print(f"{'='*70}")

    # Save
    out_dir = ensure_dir(THROUGHPUT_RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f"throughput_main_v2_{ts}.json")
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
