#!/usr/bin/env python3
"""

Hermes Agent HotpotQA Breakdown Experiment.

Implements the identical HotpotQA reasoning pipeline as DBAgent:
  1. Load N mvec vectors from reasoning_vector_test
  2. cross_encoder.pt: score all N vectors
  3. For each sample i: deberta_reader.pt -> reader_decoder.py
  4. Return decoded texts

Runs the pipeline and reports per-stage timing.
Used to compare Hermes Agent breakdown vs DBAgent Agent breakdown.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import os, sys, time, json
import numpy as np
import torch
import subprocess as sp
import psycopg2

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "morphingdb_test", "models")
TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
DB_CONFIG = {"dbname": "postgres", "host": "localhost", "port": "5432", "user": "why", "password": "123456"}


def load_mvec_vectors(n):
    """Load N mvec vectors from reasoning_vector_test."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT get_mvec_data(reasoning_vec) FROM reasoning_vector_test ORDER BY id LIMIT %s", (n,))
    vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()
    return vecs


def load_step2_vectors(n):
    """Load 2*N mvec vectors from reasoning_vector_step2."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT get_mvec_data(text_vec) FROM reasoning_vector_step2 ORDER BY id LIMIT %s", (2 * n,))
    vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()
    return vecs


def run_pipeline(n):
    """Run full HotpotQA reasoning pipeline with per-stage timing."""
    timings = {}
    results = []
    
    # Stage 1: Data Loading (mvec -> tensor conversion)
    t0 = time.perf_counter()
    input_vecs = load_mvec_vectors(n)
    step2_vecs = load_step2_vectors(n)
    # Convert to torch tensors (Perception-equivalent)
    cross_batch = torch.from_numpy(np.stack([v[:2, :] for v in input_vecs], axis=0).astype(np.float32))
    step2_tensors = []
    for i in range(n):
        idx = i * 2
        t1 = np.stack([step2_vecs[idx][:2, :], step2_vecs[idx+1][:2, :]], axis=0).astype(np.float32)
        step2_tensors.append(torch.from_numpy(t1))
    timings["data_loading_ms"] = (time.perf_counter() - t0) * 1000
    
    # Stage 2: Model Loading
    t0 = time.perf_counter()
    model_cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location="cpu")
    model_cross.eval()
    model_reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu")
    model_reader.eval()
    timings["model_loading_ms"] = (time.perf_counter() - t0) * 1000
    
    # Stage 3: Cross-Encoder Inference
    t0 = time.perf_counter()
    with torch.no_grad():
        scores = model_cross(cross_batch).squeeze(-1).tolist()
    timings["cross_encoder_ms"] = (time.perf_counter() - t0) * 1000
    
    # Stage 4: Deberta Reader + Reader Decoder (per sample)
    t0 = time.perf_counter()
    deberta_times = []
    decoder_times = []
    for i in range(n):
        # deberta_reader
        td = time.perf_counter()
        with torch.no_grad():
            reader_output = model_reader(step2_tensors[i])
        start_logits, end_logits = reader_output[:, 0, :], reader_output[:, 1, :]
        deberta_times.append((time.perf_counter() - td) * 1000)
        
        # Build res_str
        res_str_parts = []
        for j in [0, 1]:
            vec_idx = i * 2 + j
            input_ids = step2_vecs[vec_idx][0, :].astype(int).tolist()
            s = int(start_logits[j].argmax().item())
            e = int(end_logits[j].argmax().item())
            res_str_parts.append(",".join(str(t) for t in input_ids[s:e+1]))
        res_str = ";".join(res_str_parts)
        
        # reader_decoder subprocess
        td2 = time.perf_counter()
        result = sp.run(
            ["python3", f"{TOOLS_DIR}/reader_decoder.py", "--ids", res_str],
            capture_output=True, text=True, cwd=TOOLS_DIR, timeout=60
        )
        decoder_times.append((time.perf_counter() - td2) * 1000)
        results.append(result.stdout.strip())
    
    timings["deberta_reader_total_ms"] = sum(deberta_times)
    timings["reader_decoder_total_ms"] = sum(decoder_times)
    timings["num_samples"] = n
    timings["decoded_results"] = results
    
    # Summary
    total_ms = sum(v for k, v in timings.items() if k.endswith("_ms"))
    timings["total_ms"] = total_ms
    
    return timings


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print(f"Running HotpotQA pipeline for n={n}...")
    timings = run_pipeline(n)
    
    print(f"\n{'='*60}")
    print(f"HOTPOTQA PIPELINE TIMING (n={n})")
    print(f"{'='*60}")
    print(f"Data Loading:       {timings['data_loading_ms']:>10.3f} ms")
    print(f"Model Loading:      {timings['model_loading_ms']:>10.3f} ms")
    print(f"Cross-Encoder:      {timings['cross_encoder_ms']:>10.3f} ms")
    print(f"Deberta Reader:     {timings['deberta_reader_total_ms']:>10.3f} ms")
    print(f"Reader Decoder:     {timings['reader_decoder_total_ms']:>10.3f} ms")
    print(f"{'-'*60}")
    print(f"TOTAL:              {timings['total_ms']:>10.3f} ms ({timings['total_ms']/1000:.2f}s)")
    
    # Save for analysis
    with open("/tmp/hermes_hotpotqa_timings.json", "w") as f:
        json.dump(timings, f, indent=2)
    print(f"\nTimings saved to /tmp/hermes_hotpotqa_timings.json")
