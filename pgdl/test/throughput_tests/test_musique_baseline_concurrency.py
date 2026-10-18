#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import multiprocessing
from datetime import datetime

import numpy as np
import torch
import psycopg2
import subprocess as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from general_tests.benchmark_musique_fair import call_reader_decoder

MODEL_DIR = os.path.join(ROOT, "morphingdb_test", "models")
from support.test_result_paths import THROUGHPUT_RESULTS_DIR, ensure_dir

RESULTS_DIR = ensure_dir(THROUGHPUT_RESULTS_DIR)
CONCURRENCY_LEVELS = [1, 5, 10]
ROW_COUNT = 20
QUERY_TIMEOUT = 3600
COOL_DOWN = 15
WALL_TIME_LIMIT = 42.6555
TOP_K = 2
DB_CONFIG = {"dbname": "postgres", "host": "localhost", "port": "5432", "user": "why", "password": "123456"}

NEUR_INPUT_VECS = None
NEUR_STEP2_VECS = None
NEUR_MODEL_CROSS = None
NEUR_MODEL_READER = None
MORPH_STEP2_VECS = None
MORPH_MODEL_READER = None


def load_musique_mvec_vectors(n):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT get_mvec_data(reasoning_vec) FROM musique_vector_test ORDER BY id LIMIT %s", (n,))
    vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()
    return vecs


def load_musique_step2_vectors(n):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT get_mvec_data(text_vec) FROM musique_vector_step2 ORDER BY id LIMIT %s", (2 * n,))
    vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()
    return vecs


def prepare_neurdb():
    global NEUR_INPUT_VECS, NEUR_STEP2_VECS, NEUR_MODEL_CROSS, NEUR_MODEL_READER
    if NEUR_INPUT_VECS is None:
        NEUR_INPUT_VECS = load_musique_mvec_vectors(ROW_COUNT)
    if NEUR_STEP2_VECS is None:
        NEUR_STEP2_VECS = load_musique_step2_vectors(ROW_COUNT)
    if NEUR_MODEL_CROSS is None:
        NEUR_MODEL_CROSS = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location="cpu")
        NEUR_MODEL_CROSS.eval()
    if NEUR_MODEL_READER is None:
        NEUR_MODEL_READER = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu")
        NEUR_MODEL_READER.eval()


def prepare_morphingdb():
    global MORPH_STEP2_VECS, MORPH_MODEL_READER
    if MORPH_STEP2_VECS is None:
        MORPH_STEP2_VECS = load_musique_step2_vectors(ROW_COUNT)
    if MORPH_MODEL_READER is None:
        MORPH_MODEL_READER = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu")
        MORPH_MODEL_READER.eval()


def run_neurdb_once():
    batch = torch.from_numpy(np.stack([v[:2, :] for v in NEUR_INPUT_VECS], axis=0).astype(np.float32))
    with torch.no_grad():
        scores = NEUR_MODEL_CROSS(batch).squeeze(-1).tolist()
    top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:TOP_K]

    block_vecs = []
    for orig_idx, _ in top_k:
        idx = orig_idx * 2
        block_vecs.append(NEUR_STEP2_VECS[idx][:2, :])
        block_vecs.append(NEUR_STEP2_VECS[idx + 1][:2, :])
    block_batch = torch.from_numpy(np.stack(block_vecs, axis=0).astype(np.float32))
    with torch.no_grad():
        block_output = NEUR_MODEL_READER(block_batch)

    decoded_texts = []
    for i, (orig_idx, _) in enumerate(top_k):
        idx = orig_idx * 2
        start_logits = block_output[i * 2, 0, :]
        end_logits = block_output[i * 2, 1, :]
        res_str_parts = []
        for j in [0, 1]:
            vec_idx = idx + j
            input_ids = NEUR_STEP2_VECS[vec_idx][0, :].astype(int).tolist()
            s = int(start_logits.argmax().item())
            e = int(end_logits.argmax().item())
            res_str_parts.append(",".join(str(t) for t in input_ids[s:e+1]))
        decoded_texts.append(call_reader_decoder(";".join(res_str_parts)))
    return len(decoded_texts)


def run_morphingdb_once():
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT register_process()")
        cur.execute("SET statement_timeout = '1800s'")
        cross_query = (
            "SELECT predict_batch_float8('cross_encoder', 'gpu:0', reasoning_vec) "
            "OVER (ROWS BETWEEN CURRENT ROW AND 15 FOLLOWING) "
            f"FROM musique_vector_test LIMIT {ROW_COUNT};"
        )
        cur.execute(cross_query)
        rows = cur.fetchall()
        conn.close()
        conn = None
    except Exception as e:
        if conn is not None:
            try: conn.close()
            except: pass
        return {"status": "failed", "error": str(e)}

    scores = []
    for row in rows:
        val = row[0]
        if isinstance(val, (list, tuple)) and val:
            scores.append(float(val[0]))
        else:
            try:
                scores.append(float(val))
            except Exception:
                scores.append(0.0)
    top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:TOP_K]

    decoded_texts = []
    for orig_idx, _ in top_k:
        idx = orig_idx * 2
        reader_inputs = np.stack([
            MORPH_STEP2_VECS[idx][:2, :],
            MORPH_STEP2_VECS[idx + 1][:2, :],
        ], axis=0).astype(np.float32)
        reader_batch = torch.from_numpy(reader_inputs)
        with torch.no_grad():
            reader_output = MORPH_MODEL_READER(reader_batch)
        start_logits, end_logits = reader_output[:, 0, :], reader_output[:, 1, :]
        res_str_parts = []
        for j in [0, 1]:
            vec_idx = idx + j
            input_ids = MORPH_STEP2_VECS[vec_idx][0, :].astype(int).tolist()
            s = int(start_logits[j].argmax().item())
            e = int(end_logits[j].argmax().item())
            res_str_parts.append(",".join(str(t) for t in input_ids[s:e+1]))
        decoded_texts.append(call_reader_decoder(";".join(res_str_parts)))
    return {"status": "success", "rows": len(decoded_texts)}


def run_worker(task_id, system_name):
    try:
        os.setpgid(0, 0)
    except OSError:
        pass

    try:
        t0 = time.perf_counter()
        if system_name == "morphingdb":
            result = run_morphingdb_once()
            if result.get("status") != "success":
                return {"task_id": task_id, "status": "failed", "latency": None, "rows": 0, "error": result.get("error", "unknown")}
            elapsed = time.perf_counter() - t0
            return {"task_id": task_id, "status": "success", "latency": round(elapsed, 4), "rows": result.get("rows", 0), "error": None}
        if system_name == "neurdb":
            count = run_neurdb_once()
            elapsed = time.perf_counter() - t0
            return {"task_id": task_id, "status": "success", "latency": round(elapsed, 4), "rows": count, "error": None}
        return {"task_id": task_id, "status": "failed", "latency": None, "rows": 0, "error": f"unknown system: {system_name}"}
    except Exception as e:
        return {"task_id": task_id, "status": "failed", "latency": None, "rows": 0, "error": str(e)}


def test_level(system_name, concurrency):
    print(f"    Testing {concurrency:>3d} concurrent workers...", end=" ", flush=True)
    t0 = time.time()
    ctx = multiprocessing.get_context("fork")
    pool = ctx.Pool(processes=concurrency, initializer=os.setpgid, initargs=(0, 0))

    results = []
    aborted = False
    abort_reason = None
    try:
        async_results = [pool.apply_async(run_worker, args=(i, system_name)) for i in range(concurrency)]
        for i, ar in enumerate(async_results):
            elapsed = time.time() - t0
            if elapsed > WALL_TIME_LIMIT:
                aborted = True
                abort_reason = f"wall_time_limit_exceeded>{WALL_TIME_LIMIT:.0f}s"
                pool.terminate()
                results.append({"task_id": i, "status": "failed", "latency": None, "rows": 0, "error": abort_reason})
                for j in range(i + 1, concurrency):
                    results.append({"task_id": j, "status": "failed", "latency": None, "rows": 0, "error": abort_reason})
                break
            try:
                timeout = max(1, min(QUERY_TIMEOUT, WALL_TIME_LIMIT - elapsed + 1))
                results.append(ar.get(timeout=timeout))
            except Exception as e:
                results.append({"task_id": i, "status": "failed", "latency": None, "rows": 0, "error": str(e)[:200]})
    finally:
        pool.terminate()
        pool.join()

    wall = time.time() - t0
    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    avg_lat = sum(r["latency"] for r in success) / len(success) if success and not failed else None

    status_tag = f"{len(success)}/{concurrency} OK"
    if failed:
        status_tag += f", {len(failed)} FAILED"
    print(f"wall={wall:.1f}s  {status_tag}")
    for f in failed[:5]:
        print(f"      -> task {f['task_id']}: {str(f.get('error','?'))[:120]}")
    if len(failed) > 5:
        print(f"      -> ... {len(failed) - 5} more failures")

    return {
        "concurrency": concurrency,
        "wall_time": round(wall, 4),
        "total_workers": concurrency,
        "success_count": len(success),
        "fail_count": len(failed),
        "avg_latency": round(avg_lat, 4) if avg_lat is not None else None,
        "wall_time_limit": WALL_TIME_LIMIT,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "results": results,
    }


def binary_search(system_name, low_pass, high_fail):
    checked = []
    lo, hi = low_pass + 1, high_fail - 1
    best = low_pass
    while lo <= hi:
        mid = (lo + hi) // 2
        print(f"\n  --- Binary concurrency: {mid} ---")
        r = test_level(system_name, mid)
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


def run_system(system_name):
    if system_name == "neurdb":
        prepare_neurdb()
    elif system_name == "morphingdb":
        prepare_morphingdb()
    else:
        raise ValueError(system_name)

    print("=" * 70)
    print(f"  {system_name} Musique Throughput Search — {datetime.now().isoformat()}")
    print(f"  Dataset: Musique (reasoning), rows per worker={ROW_COUNT}")
    print(f"  Initial levels: {CONCURRENCY_LEVELS}")
    print(f"  Wall-time limit per level: {WALL_TIME_LIMIT:.0f}s")
    print("=" * 70)

    all_results = []
    pass_levels = []
    fail_levels = []
    stop_early = False
    for idx, level in enumerate(CONCURRENCY_LEVELS):
        print(f"\n  --- Concurrency: {level} ---")
        r = test_level(system_name, level)
        all_results.append(r)
        if r["fail_count"] == 0:
            pass_levels.append(level)
        else:
            fail_levels.append(level)
            if level == 1:
                stop_early = True
                break
        if idx < len(CONCURRENCY_LEVELS) - 1 and not stop_early:
            print(f"    Cooling {COOL_DOWN}s...")
            time.sleep(COOL_DOWN)

    max_pass = max(pass_levels) if pass_levels else 0
    min_fail = min(fail_levels) if fail_levels else None
    binary_results = []
    if max_pass and min_fail and min_fail - max_pass > 1:
        best, binary_results = binary_search(system_name, max_pass, min_fail)
        max_pass = best
        all_results.extend(binary_results)

    all_results = sorted(all_results, key=lambda x: x["concurrency"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"{system_name}_musique_concurrency_{ts}.json")
    payload = {
        "timestamp": datetime.now().isoformat(),
        "system": system_name,
        "dataset": "musique_reasoning",
        "row_count": ROW_COUNT,
        "wall_time_limit": WALL_TIME_LIMIT,
        "max_passing_concurrency": max_pass,
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")
    print(f"Max passing concurrency: {max_pass}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["morphingdb", "neurdb"], required=True)
    args = parser.parse_args()
    run_system(args.system)


if __name__ == "__main__":
    main()
