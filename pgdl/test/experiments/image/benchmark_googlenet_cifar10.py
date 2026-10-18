#!/usr/bin/env python3
"""
Load GoogLeNet CIFAR-10 model and run batched inference on 1000 image vectors from PostgreSQL.

- Model: /home/why/baseline/models/googlenet_cifar10.pt (TorchScript)
- Data: cifar_image_vector_table (mvec type, preprocessed 1x3x224x224 float32)
- Batch size: 32
"""

import time
import struct
import psycopg2
import torch
import numpy as np

MODEL_PATH = "/home/why/baseline/models/googlenet_cifar10.pt"
DB_CONFIG = {
    "host": "localhost",
    "user": "why",
    "password": "123456",
    "dbname": "postgres",
}
BATCH_SIZE = 32
NUM_SAMPLES = 1000

# ---- Step 1: Load model on GPU ----
print("Loading model on cuda:0 ...")
device = torch.device("cuda:0")
model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()
print(f"  Model loaded: {MODEL_PATH}")

# ---- Step 2: Fetch 1000 image vectors from PostgreSQL ----
print(f"Fetching {NUM_SAMPLES} image vectors from cifar_image_vector_table ...")
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
cur.execute(
    "SELECT id, get_mvec_shape(image_vector)::int[], get_mvec_data(image_vector) "
    "FROM cifar_image_vector_table LIMIT %s;",
    (NUM_SAMPLES,)
)
rows = cur.fetchall()
cur.close()
conn.close()
print(f"  Fetched {len(rows)} rows")

# ---- Step 3: Convert mvec data to tensors ----
print("Converting mvec data to batched GPU tensors ...")
tensors = []
for row_id, shape, data in rows:
    # shape is e.g. {1,3,224,224}, data is a float4[] array
    arr = np.array(data, dtype=np.float32)
    t = torch.from_numpy(arr).view(*shape).to(device)
    tensors.append(t)

# Stack into a single batch tensor: (N, C, H, W)
batch_tensor = torch.cat(tensors, dim=0)  # (1000, 3, 224, 224)
print(f"  Input tensor shape: {batch_tensor.shape}, dtype: {batch_tensor.dtype}")

# ---- Step 4: Warm-up run ----
print("Warm-up inference (1 batch) ...")
with torch.no_grad():
    _ = model(batch_tensor[:BATCH_SIZE])
torch.cuda.synchronize(device)
print("  Warm-up complete.")

# ---- Step 5: Batched inference with timing ----
print(f"Running batched inference (batch_size={BATCH_SIZE}, {len(tensors)} samples) ...")
num_batches = (len(tensors) + BATCH_SIZE - 1) // BATCH_SIZE
all_outputs = []

torch.cuda.synchronize(device)
t_start = time.perf_counter()

with torch.no_grad():
    for i in range(num_batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(tensors))
        batch = batch_tensor[start:end]
        output = model(batch)
        all_outputs.append(output)

torch.cuda.synchronize(device)
t_end = time.perf_counter()

inference_time = t_end - t_start
total_outputs = torch.cat(all_outputs, dim=0)

print(f"\n{'='*60}")
print(f"  MODEL:        googlenet_cifar10 (TorchScript)")
print(f"  DEVICE:       {device}")
print(f"  SAMPLES:      {len(tensors)}")
print(f"  BATCH SIZE:   {BATCH_SIZE}")
print(f"  NUM BATCHES:  {num_batches}")
print(f"  INPUT SHAPE:  {batch_tensor.shape}")
print(f"  OUTPUT SHAPE: {total_outputs.shape}")
print(f"  INFERENCE:    {inference_time*1000:.2f} ms ({inference_time:.4f} s)")
print(f"  PER SAMPLE:   {inference_time/len(tensors)*1000:.3f} ms")
print(f"  THROUGHPUT:   {len(tensors)/inference_time:.1f} samples/sec")
print(f"{'='*60}")
