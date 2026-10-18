#!/usr/bin/env python3
"""
Correctness Comparison: DBAgent vs Hermes vs Python Baseline on HotpotQA
=========================================================================
Evaluates whether agent frameworks introduce errors into model outputs.
Method: Compare decoded answers against pure-Python baseline (ground truth).

n values: 1, 5, 10, 20
"""

import os, sys, time, json, subprocess as sp
import numpy as np
import torch
import psycopg2
from datetime import datetime
from tools.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "morphingdb_test", "models")
TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
PSQL = "/home/why/dbagent/pg_base/bin/psql"
DB_CONFIG = {"dbname": "postgres", "host": "localhost", "port": "5432",
             "user": "why", "password": "123456"}

N_VALUES = [1, 5, 10, 20]


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════
def load_mvec_vectors(n):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT get_mvec_data(reasoning_vec) FROM reasoning_vector_test ORDER BY id LIMIT %s", (n,))
    vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()
    return vecs

def load_step2_vectors(n):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT get_mvec_data(text_vec) FROM reasoning_vector_step2 ORDER BY id LIMIT %s", (2 * n,))
    vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()
    return vecs

def call_reader_decoder(res_str):
    result = sp.run(
        ["uv", "run", f"{TOOLS_DIR}/reader_decoder.py", "--ids", res_str],
        capture_output=True, text=True, cwd=TOOLS_DIR, timeout=60
    )
    return result.stdout.strip()


# ═══════════════════════════════════════════════════════════════════════
# 1. Pure Python Baseline (ground truth)
# ═══════════════════════════════════════════════════════════════════════
def run_python_baseline(n, input_vecs, step2_vecs):
    """Pure Python: cross_encoder -> deberta_reader -> reader_decoder.
    Returns list of decoded texts."""
    model_cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location="cpu")
    model_cross.eval()
    model_reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu")
    model_reader.eval()

    # Stage 1: cross_encoder
    batch = torch.from_numpy(np.stack([v[:2, :] for v in input_vecs], axis=0).astype(np.float32))
    with torch.no_grad():
        scores = model_cross(batch).squeeze(-1).tolist()

    # Stage 2: deberta_reader + reader_decoder per sample
    decoded_texts = []
    for i in range(n):
        idx = i * 2
        reader_inputs = np.stack([step2_vecs[idx][:2, :], step2_vecs[idx+1][:2, :]], axis=0).astype(np.float32)
        reader_batch = torch.from_numpy(reader_inputs)
        with torch.no_grad():
            reader_output = model_reader(reader_batch)
        start_logits, end_logits = reader_output[:, 0, :], reader_output[:, 1, :]

        res_str_parts = []
        for j in [0, 1]:
            vec_idx = idx + j
            input_ids = step2_vecs[vec_idx][0, :].astype(int).tolist()
            s = int(start_logits[j].argmax().item())
            e = int(end_logits[j].argmax().item())
            res_str_parts.append(",".join(str(t) for t in input_ids[s:e+1]))
        res_str = ";".join(res_str_parts)
        decoded = call_reader_decoder(res_str)
        decoded_texts.append(decoded)

    return decoded_texts


# ═══════════════════════════════════════════════════════════════════════
# 2. DBAgent (db_agent_single via psql)
# ═══════════════════════════════════════════════════════════════════════
def run_dbagent(n):
    """DBAgent via db_agent_single SQL. Returns list of decoded texts."""
    sql = f"SELECT unnest(db_agent_single('reasoning', sub_table.reasoning_vec)) AS score FROM (SELECT * FROM reasoning_vector_test ORDER BY id LIMIT {n}) AS sub_table;"
    cmd = [PSQL, "-h", "localhost", "-U", "why", "-d", "postgres", "-c", sql]
    env = os.environ.copy()
    env["PGPASSWORD"] = "123456"

    result = sp.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    # psql returns non-zero for INFO/NOTICE messages — check for actual errors
    stderr_upper = (result.stderr or '').upper()
    if 'ERROR' in stderr_upper or 'FATAL' in stderr_upper:
        print(f"    DBAgent FAILED: {result.stderr[:200]}")
        return None

    # Parse stdout: each non-header line is an answer (JSON object)
    stdout = result.stdout.strip()
    texts = []
    # Find JSON objects in output: each starts with '{' and ends with '}'
    depth = 0
    buf = []
    for line in stdout.split('\n'):
        stripped = line.strip()
        if stripped.startswith('{'):
            depth = 1
            buf = [stripped]
        elif depth > 0:
            buf.append(stripped)
            if '}' in stripped:
                depth -= 1
                if depth == 0:
                    texts.append('\n'.join(buf))
                    buf = []
    if not texts:
        # Fallback: treat each non-psql-header line as an answer
        for line in stdout.split('\n'):
            s = line.strip()
            if s and not s.startswith('score') and not s.startswith('---') and 'unnest' not in s.lower():
                if s.startswith('{'):
                    texts.append(s)
    print(f"    Parsed {len(texts)} answers from stdout")
    return texts if texts else None


# ═══════════════════════════════════════════════════════════════════════
# 3. Hermes (Python pipeline matching hermes benchmark pattern)
# ═══════════════════════════════════════════════════════════════════════
def run_hermes(n, input_vecs, step2_vecs):
    """Hermes-style: batched deberta_reader. Returns list of decoded texts."""
    model_cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location="cpu")
    model_cross.eval()
    model_reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu")
    model_reader.eval()

    # Stage 1: cross_encoder
    batch = torch.from_numpy(np.stack([v[:2, :] for v in input_vecs], axis=0).astype(np.float32))
    with torch.no_grad():
        scores = model_cross(batch).squeeze(-1).tolist()

    # Stage 2: deberta_reader batched ALL samples, then reader_decoder per sample
    all_step2 = []
    for i in range(n):
        idx = i * 2
        all_step2.append(step2_vecs[idx][:2, :])
        all_step2.append(step2_vecs[idx + 1][:2, :])
    reader_batch = torch.from_numpy(np.stack(all_step2, axis=0).astype(np.float32))
    with torch.no_grad():
        reader_output = model_reader(reader_batch)

    decoded_texts = []
    for i in range(n):
        idx = i * 2
        start_logits = reader_output[i*2, 0, :]
        end_logits = reader_output[i*2, 1, :]
        res_str_parts = []
        for j in [0, 1]:
            vec_idx = idx + j
            input_ids = step2_vecs[vec_idx][0, :].astype(int).tolist()
            s = int(start_logits.argmax().item())
            e = int(end_logits.argmax().item())
            res_str_parts.append(",".join(str(t) for t in input_ids[s:e+1]))
        res_str = ";".join(res_str_parts)
        decoded = call_reader_decoder(res_str)
        decoded_texts.append(decoded)

    return decoded_texts


# ═══════════════════════════════════════════════════════════════════════
# Comparison
# ═══════════════════════════════════════════════════════════════════════
import re

def strip_timing(text):
    """Remove 'Execution time: X.XXXX seconds' line from reader_decoder output."""
    return re.sub(r'\n?Execution time: [\d.]+ seconds?\n?', '', text).strip()

def compare_answers(baseline, target, name):
    """Compare target answers against baseline.
    Strips timing noise from reader_decoder before comparison."""
    if target is None:
        return {"name": name, "match_count": 0, "total": len(baseline),
                "match_rate": 0.0, "status": "FAILED"}
    if len(target) != len(baseline):
        print(f"    WARNING: {name} returned {len(target)} answers, baseline has {len(baseline)}")

    matches = 0
    details = []
    n_cmp = min(len(baseline), len(target))
    for i in range(n_cmp):
        b_clean = strip_timing(baseline[i])
        t_clean = strip_timing(target[i])
        is_match = (b_clean == t_clean)
        if is_match:
            matches += 1
        details.append({"index": i, "match": is_match,
                        "baseline_preview": b_clean[:120],
                        "target_preview": t_clean[:120]})
    return {
        "name": name,
        "match_count": matches,
        "total": n_cmp,
        "match_rate": matches / n_cmp if n_cmp else 0.0,
        "details": details,
        "status": "OK"
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print(f"  CORRECTNESS TEST — {datetime.now().isoformat()}")
    print(f"  HotpotQA: DBAgent vs Hermes vs Python Baseline")
    print(f"  n values: {N_VALUES}")
    print("=" * 70)

    all_results = {}

    for n in N_VALUES:
        print(f"\n{'#'*60}")
        print(f"# n = {n}")
        print(f"{'#'*60}")

        # Load data once
        print("  Loading data...")
        input_vecs = load_mvec_vectors(n)
        step2_vecs = load_step2_vectors(n)
        print(f"  Loaded {len(input_vecs)} mvec + {len(step2_vecs)} step2 vectors")

        # 1. Python baseline (ground truth)
        print("  [1/3] Python baseline...")
        t0 = time.perf_counter()
        baseline = run_python_baseline(n, input_vecs, step2_vecs)
        bt = time.perf_counter() - t0
        print(f"    {len(baseline)} answers in {bt:.1f}s")
        for i, ans in enumerate(baseline):
            print(f"    [{i}] {ans[:120]}")

        # 2. DBAgent
        print("  [2/3] DBAgent...")
        t0 = time.perf_counter()
        dbagent_texts = run_dbagent(n)
        dt = time.perf_counter() - t0
        if dbagent_texts:
            print(f"    {len(dbagent_texts)} answers in {dt:.1f}s")
            for i, ans in enumerate(dbagent_texts[:n]):
                print(f"    [{i}] {ans[:120]}")
        else:
            print(f"    FAILED")

        # 3. Hermes
        print("  [3/3] Hermes...")
        t0 = time.perf_counter()
        hermes_texts = run_hermes(n, input_vecs, step2_vecs)
        ht = time.perf_counter() - t0
        print(f"    {len(hermes_texts)} answers in {ht:.1f}s")
        for i, ans in enumerate(hermes_texts):
            print(f"    [{i}] {ans[:120]}")

        # Compare
        dbagent_cmp = compare_answers(baseline, dbagent_texts, "DBAgent")
        hermes_cmp = compare_answers(baseline, hermes_texts, "Hermes")

        all_results[n] = {
            "baseline": baseline,
            "dbagent": dbagent_texts,
            "hermes": hermes_texts,
            "dbagent_comparison": dbagent_cmp,
            "hermes_comparison": hermes_cmp,
        }

        print(f"\n  Results for n={n}:")
        print(f"    DBAgent: {dbagent_cmp['match_count']}/{dbagent_cmp['total']} "
              f"({dbagent_cmp['match_rate']:.1%})")
        print(f"    Hermes:  {hermes_cmp['match_count']}/{hermes_cmp['total']} "
              f"({hermes_cmp['match_rate']:.1%})")

        # Cooldown
        time.sleep(5)

    # Final summary
    print(f"\n{'='*70}")
    print("  CORRECTNESS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'n':<6} {'DBAgent':>12} {'Hermes':>12}")
    print(f"  {'-'*30}")
    for n in N_VALUES:
        d = all_results[n]['dbagent_comparison']['match_rate']
        h = all_results[n]['hermes_comparison']['match_rate']
        print(f"  {n:<6} {d:>11.1%} {h:>11.1%}")
    print(f"{'='*70}")

    # Save
    out_dir = ensure_dir(GENERAL_RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f"correctness_{ts}.json")
    # Simplify for JSON
    save_data = {}
    for n in N_VALUES:
        dc = all_results[n]['dbagent_comparison']
        hc = all_results[n]['hermes_comparison']
        save_data[str(n)] = {
            "dbagent_match_rate": dc['match_rate'],
            "hermes_match_rate": hc['match_rate'],
            "dbagent_matches": dc['match_count'],
            "hermes_matches": hc['match_count'],
            "total": dc['total'],
            "dbagent_status": dc['status'],
            "hermes_status": hc['status'],
        }
    with open(path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
