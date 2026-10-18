#!/usr/bin/env python3
"""Idle HotpotQA n=20 breakdown, aligned to the main-table totals.

DBAgent: one persistent connection, register_process once, discard one n=1
warmup (pays model load), time n=20. Categories come from the backend log:
Model Execution is the cross-encoder plus the chunked deberta reader,
Answer Materialization is the reader_decoder subprocesses, Orchestration is
the remainder of the wall time.

Hermes: the same pipeline the main-table prompt specifies, run directly so the
split is measured rather than generated. Both models load on GPU before the
timer. One discarded warmup, then one timed run. Model Execution is the two
forwards, Answer Materialization is the uv-run decoder, Orchestration is data
loading and span assembly.

Decoder calls go through `uv run`, the interpreter the main table uses.
"""
import os
import re
import sys
import json
import time
import subprocess
from datetime import datetime

import numpy as np
import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from support.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "morphingdb_test", "models")
TOOLS_DIR = os.path.join(ROOT, "tools")
DB = {"dbname": "postgres", "host": "localhost", "port": "5432",
      "user": "why", "password": "123456"}
LOG = "/home/why/dbagent/pg_base/data/log/postgresql-%s.log" % datetime.now().strftime("%Y-%m-%d")
N = 20
PAPER = {"DBAgent": 59.5697, "Hermes": 69.8286}


def query(cur, n):
    cur.execute(
        "SELECT unnest(db_agent_single('reasoning', sub_table.reasoning_vec)) "
        "AS score FROM (SELECT * FROM reasoning_vector_test ORDER BY id "
        f"LIMIT {n}) AS sub_table;")
    return cur.fetchall()


def log_len():
    return os.path.getsize(LOG)


def read_new(start):
    with open(LOG, "rb") as f:
        f.seek(start)
        return f.read().decode("utf-8", "replace")


def run_dbagent():
    conn = psycopg2.connect(**DB)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT register_process();")

    t = time.perf_counter()
    warm_rows = query(cur, 1)
    warmup_s = time.perf_counter() - t

    start = log_len()
    t = time.perf_counter()
    rows = query(cur, N)
    wall_ms = (time.perf_counter() - t) * 1000
    text = read_new(start)
    conn.close()

    step2 = [int(x) for x in re.findall(r"chunked STEP2 execution took (\d+) ms", text)]
    dec = [int(x) for x in re.findall(r"reader_decoder for sample \d+ took (\d+) ms", text)]
    agents = dict(re.findall(r"Agent:\s+(\S+)\s+Total:\s+([0-9.]+) ms", text))

    material = float(sum(dec))
    model = float(sum(step2))
    orchestration = wall_ms - model - material
    return {
        "warmup_s": round(warmup_s, 4),
        "warmup_rows": len(warm_rows),
        "rows": len(rows),
        "wall_ms": round(wall_ms, 3),
        "model_execution_ms": round(model, 3),
        "answer_materialization_ms": round(material, 3),
        "orchestration_ms": round(orchestration, 3),
        "step2_chunks_ms": step2,
        "decoder_calls": len(dec),
        "agents_ms": agents,
    }


def run_hermes():
    import torch

    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("SELECT get_mvec_data(reasoning_vec) FROM reasoning_vector_test "
                "ORDER BY id LIMIT %s", (N,))
    inputs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    cur.execute("SELECT get_mvec_data(text_vec) FROM reasoning_vector_step2 "
                "ORDER BY id LIMIT %s", (2 * N,))
    step2 = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()

    gpu = torch.device("cuda:0")
    t = time.perf_counter()
    cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location=gpu).eval()
    reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu").eval()
    load_ms = (time.perf_counter() - t) * 1000

    def once():
        t0 = time.perf_counter()
        t1 = time.perf_counter()
        batch = torch.from_numpy(
            np.stack([v[:2, :] for v in inputs]).astype(np.float32)).to(gpu)
        spans = []
        for i in range(N):
            arr = np.stack([step2[i * 2][:2, :], step2[i * 2 + 1][:2, :]]).astype(np.float32)
            spans.append(torch.from_numpy(arr))
        data_ms = (time.perf_counter() - t1) * 1000

        t1 = time.perf_counter()
        with torch.no_grad():
            scores = cross(batch)
        torch.cuda.synchronize()
        cross_ms = (time.perf_counter() - t1) * 1000
        t1 = time.perf_counter()
        with torch.no_grad():
            outs = [reader(s) for s in spans]
        deberta_ms = (time.perf_counter() - t1) * 1000
        model_ms = cross_ms + deberta_ms

        t1 = time.perf_counter()
        decoded = 0
        for i, out in enumerate(outs):
            sl, el = out[:, 0, :], out[:, 1, :]
            parts = []
            for j in (0, 1):
                ids = step2[i * 2 + j][0, :].astype(int).tolist()
                s, e = int(sl[j].argmax()), int(el[j].argmax())
                parts.append(",".join(str(x) for x in ids[s:e + 1]))
            subprocess.run(
                ["uv", "run", f"{TOOLS_DIR}/reader_decoder.py", "--ids", ";".join(parts)],
                capture_output=True, text=True, cwd=TOOLS_DIR, timeout=120, check=False)
            decoded += 1
        material_ms = (time.perf_counter() - t1) * 1000
        wall_ms = (time.perf_counter() - t0) * 1000
        return {
            "wall_ms": round(wall_ms, 3),
            "model_execution_ms": round(model_ms, 3),
            "answer_materialization_ms": round(material_ms, 3),
            "orchestration_ms": round(wall_ms - model_ms - material_ms, 3),
            "data_loading_ms": round(data_ms, 3),
            "rows": decoded,
            "scores": int(scores.shape[0]),
            "cross_ms": round(cross_ms, 3),
            "deberta_ms": round(deberta_ms, 3),
        }

    print("hermes warmup", flush=True)
    warm = once()
    print(f"hermes warmup wall {warm['wall_ms']:.1f} ms", flush=True)
    timed = once()
    timed["warmup_wall_ms"] = warm["wall_ms"]
    timed["model_load_excluded_ms"] = round(load_ms, 1)
    return timed


def show(name, part):
    wall = part["wall_ms"] / 1000
    paper = PAPER[name]
    print(f"\n{name}: wall {wall:.3f}s  paper {paper:.3f}s  delta {wall - paper:+.3f}s")
    for key in ("model_execution_ms", "answer_materialization_ms", "orchestration_ms"):
        ms = part[key]
        print(f"  {key[:-3]:<24} {ms:10.1f} ms   {ms / part['wall_ms'] * 100:5.1f}%")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    result = {
        "timestamp": datetime.now().isoformat(),
        "load": "idle, no contention, GPU 0",
        "n": N,
        "models_loaded_before_timing": True,
        "decoder": "uv run -> pgdl/test/.venv",
        "paper_s": PAPER,
    }
    if which in ("both", "dbagent"):
        print("=== DBAgent idle n=20 ===", flush=True)
        result["dbagent"] = run_dbagent()
        show("DBAgent", result["dbagent"])
        print("decoder calls", result["dbagent"]["decoder_calls"],
              "step2 chunks", result["dbagent"]["step2_chunks_ms"])
    if which in ("both", "hermes"):
        print("\n=== Hermes idle n=20 ===", flush=True)
        result["hermes"] = run_hermes()
        show("Hermes", result["hermes"])
    path = os.path.join(ensure_dir(GENERAL_RESULTS_DIR),
                        f"hotpotqa_idle_breakdown_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print("\nsaved", path)


if __name__ == "__main__":
    main()
