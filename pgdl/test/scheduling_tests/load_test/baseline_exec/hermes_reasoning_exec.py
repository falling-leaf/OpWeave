#!/usr/bin/env python3
"""
Hermes-generated HotpotQA (reasoning) inference script (execution phase only).

Replicates the DBAgent reasoning pipeline:
  1. Load N reasoning vectors (1x4x128) from reasoning_vector_test
  2. cross_encoder.pt on GPU: score all N
  3. per-sample deberta_reader.pt on GPU -> reader_decoder.py (CPU subprocess)

Model loading is EXCLUDED from timing; data loading + inference + decode are timed.
Outputs a JSON line: {"inference_ms":..., "total_ms":..., "cross_ms":...,
                      "deberta_ms":..., "decoder_ms":..., "rows":...}

Usage:
    python hermes_reasoning_exec.py 20
"""
import os
import sys
import time
import json
import numpy as np
import torch
import subprocess as sp
import psycopg2

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "morphingdb_test", "models")
TOOLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "tools")
DB_CONFIG = {"dbname": "postgres", "host": "localhost", "port": "5432",
             "user": "why", "password": "123456"}

row_count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
# cross_encoder runs on GPU (load-sensitive); deberta_reader is pinned to CPU
# because its TorchScript trace contains a CPU-bound index tensor (same reason
# DBAgent hardcodes deberta to CPU). reader_decoder is always a CPU subprocess.
gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")


def load_mvec(table, n):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(f"SELECT get_mvec_data({'reasoning_vec' if table=='reasoning_vector_test' else 'text_vec'}) "
                f"FROM {table} ORDER BY id LIMIT %s", (n,))
    vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()
    return vecs


# Model loading (EXCLUDED from timing)
model_cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location=gpu_device)
model_cross.eval()
model_reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location=cpu_device)
model_reader.eval()

# ---- Timed: data loading ----
t0 = time.perf_counter()
input_vecs = load_mvec("reasoning_vector_test", row_count)
step2_vecs = load_mvec("reasoning_vector_step2", 2 * row_count)

cross_batch = torch.from_numpy(
    np.stack([v[:2, :] for v in input_vecs], axis=0).astype(np.float32)).to(gpu_device)
step2_tensors = []
for i in range(row_count):
    idx = i * 2
    t1 = np.stack([step2_vecs[idx][:2, :], step2_vecs[idx + 1][:2, :]], axis=0).astype(np.float32)
    step2_tensors.append(torch.from_numpy(t1).to(cpu_device))
load_ms = (time.perf_counter() - t0) * 1000

# Warmup (excluded)
with torch.no_grad():
    _ = model_cross(cross_batch)
    _ = model_reader(step2_tensors[0])
if gpu_device.type == 'cuda':
    torch.cuda.synchronize()

# ---- Timed: cross_encoder (GPU) ----
t0 = time.perf_counter()
with torch.no_grad():
    scores = model_cross(cross_batch).squeeze(-1).tolist()
if gpu_device.type == 'cuda':
    torch.cuda.synchronize()
cross_ms = (time.perf_counter() - t0) * 1000

# ---- Timed: deberta_reader (CPU, per sample) + reader_decoder (CPU subprocess) ----
deberta_ms = 0.0
decoder_ms = 0.0
results = []
for i in range(row_count):
    td = time.perf_counter()
    with torch.no_grad():
        reader_output = model_reader(step2_tensors[i])
    start_logits, end_logits = reader_output[:, 0, :], reader_output[:, 1, :]
    deberta_ms += (time.perf_counter() - td) * 1000

    res_str_parts = []
    for j in [0, 1]:
        vec_idx = i * 2 + j
        input_ids = step2_vecs[vec_idx][0, :].astype(int).tolist()
        s = int(start_logits[j].argmax().item())
        e = int(end_logits[j].argmax().item())
        res_str_parts.append(",".join(str(t) for t in input_ids[s:e + 1]))
    res_str = ";".join(res_str_parts)

    td2 = time.perf_counter()
    result = sp.run(["uv", "run", "--project", os.path.dirname(TOOLS_DIR), "python",
                     f"{TOOLS_DIR}/reader_decoder.py", "--ids", res_str],
                    capture_output=True, text=True, cwd=TOOLS_DIR, timeout=60)
    decoder_ms += (time.perf_counter() - td2) * 1000
    results.append(result.stdout.strip())

inference_ms = cross_ms + deberta_ms + decoder_ms
total_ms = load_ms + inference_ms

print(json.dumps({
    "inference_ms": round(inference_ms, 1),
    "total_ms": round(total_ms, 1),
    "load_ms": round(load_ms, 1),
    "cross_ms": round(cross_ms, 1),
    "deberta_ms": round(deberta_ms, 1),
    "decoder_ms": round(decoder_ms, 1),
    "rows": row_count,
    "device": f"cross={gpu_device},deberta={cpu_device}",
}))

