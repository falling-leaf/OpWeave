#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import multiprocessing
import subprocess as sp
from datetime import datetime

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from general_tests.benchmark_reasoning import load_mvec_vectors, load_step2_vectors, call_reader_decoder

CPU_SET = list(range(20))
CPU_MASK = ",".join(str(c) for c in CPU_SET)
MODEL_DIR = os.path.join(ROOT, "morphingdb_test", "models")
from support.test_result_paths import THROUGHPUT_RESULTS_DIR, ensure_dir

RESULTS_DIR = ensure_dir(THROUGHPUT_RESULTS_DIR)
HERMES_EXEC_SCRIPT = os.path.join(ROOT, "scheduling_tests", "baseline_code", "hermes_reasoning_exec.py")
CONCURRENCY_LEVELS = [1, 5, 10]
ROW_COUNT = 20
QUERY_TIMEOUT = 3600
COOL_DOWN = 15
WALL_TIME_LIMIT = 110.0
HERMES_EXEC_TIMEOUT = 900

G_INPUT_VECS = None
G_STEP2_VECS = None
G_MODEL_CROSS = None
G_MODEL_READER = None


def pin_current_process():
    try:
        os.sched_setaffinity(0, set(CPU_SET))
    except Exception:
        pass


def prepare_gendb():
    global G_INPUT_VECS, G_STEP2_VECS, G_MODEL_CROSS, G_MODEL_READER
    pin_current_process()
    if G_INPUT_VECS is None:
        G_INPUT_VECS = load_mvec_vectors(ROW_COUNT)
    if G_STEP2_VECS is None:
        G_STEP2_VECS = load_step2_vectors(ROW_COUNT)
    if G_MODEL_CROSS is None:
        G_MODEL_CROSS = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location="cpu")
        G_MODEL_CROSS.eval()
    if G_MODEL_READER is None:
        G_MODEL_READER = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu")
        G_MODEL_READER.eval()


def run_gendb_exec_once():
    batch = torch.from_numpy(np.stack([v[:2, :] for v in G_INPUT_VECS], axis=0).astype(np.float32))
    with torch.no_grad():
        _ = G_MODEL_CROSS(batch).squeeze(-1).tolist()

    decoded = 0
    for i in range(ROW_COUNT):
        idx = i * 2
        reader_inputs = np.stack([
            G_STEP2_VECS[idx][:2, :],
            G_STEP2_VECS[idx + 1][:2, :],
        ], axis=0).astype(np.float32)
        reader_batch = torch.from_numpy(reader_inputs)
        with torch.no_grad():
            reader_output = G_MODEL_READER(reader_batch)
        start_logits, end_logits = reader_output[:, 0, :], reader_output[:, 1, :]
        res_str_parts = []
        for j in [0, 1]:
            vec_idx = idx + j
            input_ids = G_STEP2_VECS[vec_idx][0, :].astype(int).tolist()
            s = int(start_logits[j].argmax().item())
            e = int(end_logits[j].argmax().item())
            res_str_parts.append(",".join(str(t) for t in input_ids[s:e+1]))
        call_reader_decoder(";".join(res_str_parts))
        decoded += 1
    return decoded


def run_hermes_exec_once():
    cmd = ["taskset", "-c", CPU_MASK, sys.executable, HERMES_EXEC_SCRIPT, str(ROW_COUNT)]
    result = sp.run(cmd, capture_output=True, text=True, timeout=HERMES_EXEC_TIMEOUT)
    if result.returncode != 0:
        return {"status": "failed", "error": (result.stderr or result.stdout)[-200:]}

    json_line = None
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            json_line = line
    if not json_line:
        return {"status": "failed", "error": "no JSON output"}

    data = json.loads(json_line)
    return {"status": "success", "rows": data.get("rows", ROW_COUNT), "payload": data}


def run_worker(task_id, system_name):
    try:
        os.setpgid(0, 0)
    except OSError:
        pass
    pin_current_process()

    try:
        t0 = time.perf_counter()
        if system_name == "gendb":
            rows = run_gendb_exec_once()
            elapsed = time.perf_counter() - t0
            return {"task_id": task_id, "status": "success", "latency": round(elapsed, 4), "rows": rows, "error": None}
        if system_name == "hermes":
            result = run_hermes_exec_once()
            if result["status"] != "success":
                return {"task_id": task_id, "status": "failed", "latency": None, "rows": 0, "error": result.get("error", "unknown")}
            elapsed = time.perf_counter() - t0
            return {"task_id": task_id, "status": "success", "latency": round(elapsed, 4), "rows": result.get("rows", ROW_COUNT), "error": None}
        return {"task_id": task_id, "status": "failed", "latency": None, "rows": 0, "error": f"unknown system: {system_name}"}
    except Exception as e:
        return {"task_id": task_id, "status": "failed", "latency": None, "rows": 0, "error": str(e)}


def test_level(system_name, concurrency):
    print(f"    Testing {concurrency:>3d} concurrent workers...", end=" ", flush=True)
    t0 = time.time()
    ctx = multiprocessing.get_context("fork")
    pool = ctx.Pool(processes=concurrency, initializer=pin_current_process)

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
    pin_current_process()
    if system_name == "gendb":
        prepare_gendb()

    print("=" * 70)
    print(f"  {system_name} HotpotQA Exec-Only Throughput Search — {datetime.now().isoformat()}")
    print(f"  Dataset: HotpotQA (reasoning), rows per worker={ROW_COUNT}")
    print(f"  CPU affinity: {CPU_MASK}")
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
    out_path = os.path.join(RESULTS_DIR, f"{system_name}_hotpotqa_exec_concurrency_{ts}.json")
    payload = {
        "timestamp": datetime.now().isoformat(),
        "system": system_name,
        "mode": "exec_only",
        "dataset": "hotpotqa_reasoning",
        "row_count": ROW_COUNT,
        "cpu_affinity": CPU_SET,
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
    parser.add_argument("--system", choices=["gendb", "hermes"], required=True)
    args = parser.parse_args()
    run_system(args.system)


if __name__ == "__main__":
    main()
