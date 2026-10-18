#!/usr/bin/env python3
"""
Hermes-generated cifar10 inference script (execution phase only, GPU).

This script represents the hermes-generated execution code for cifar10.
It loads googlenet_cifar10 on GPU 0, fetches data from DB, runs batched inference.
Model loading excluded from timing; data loading and inference are both timed.

Outputs a JSON line to stdout: {"inference_ms": ..., "total_ms": ..., "rows": ...}

Usage:
    python hermes_cifar10_exec.py 1000
"""
import torch
import time
import json
import sys
import os
import numpy as np
import psycopg2

MODEL_PATH = "/home/why/baseline/models/googlenet_cifar10.pt"
DB_CONFIG = {
    "host": "localhost", "port": "5432", "user": "why",
    "password": "123456", "dbname": "postgres",
}

row_count = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()

# Load data (timed — part of execution)
t0 = time.perf_counter()

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
cur.execute("SELECT get_mvec_data(image_vector) FROM cifar_image_vector_table LIMIT %s", (row_count,))
rows = cur.fetchall()
conn.close()

arrays = [np.array(row[0], dtype=np.float32).reshape(3, 224, 224) for row in rows]

data_array = np.stack(arrays)
data_tensor = torch.from_numpy(data_array).to(device)
load_time = time.perf_counter() - t0

BATCH_SIZE = 32
n = data_tensor.shape[0]

# Warmup
with torch.no_grad():
    _ = model(data_tensor[:BATCH_SIZE])
if device.type == 'cuda':
    torch.cuda.synchronize()

# Timed inference
start = time.perf_counter()
with torch.no_grad():
    for i in range(0, n, BATCH_SIZE):
        batch = data_tensor[i:i + BATCH_SIZE]
        _ = model(batch)
if device.type == 'cuda':
    torch.cuda.synchronize()
end = time.perf_counter()

inference_ms = (end - start) * 1000
total_ms = (end - t0) * 1000

print(json.dumps({
    "inference_ms": round(inference_ms, 2),
    "total_ms": round(total_ms, 2),
    "rows": n,
    "device": str(device),
}))
