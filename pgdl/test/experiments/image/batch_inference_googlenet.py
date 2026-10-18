"""Batch inference: load googlenet_cifar10.pt via torch.jit.load, fetch 1000 mvec
vectors from cifar_image_vector_table, run batched inference on GPU (cuda:0)."""

import time
import torch
import psycopg2
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH = "/home/why/baseline/models/googlenet_cifar10.pt"
DEVICE = "cuda:0"
BATCH_SIZE = 32
DB_CONFIG = {
    "host": "localhost",
    "user": "why",
    "password": "123456",
    "dbname": "postgres",
}
ROW_COUNT = 1000

# ── Load model ─────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}")

# ── Fetch image vectors from PostgreSQL ────────────────────────────────────────
print(f"Fetching {ROW_COUNT} image vectors from cifar_image_vector_table...")
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

cur.execute(f"""
    SELECT get_mvec_data(image_vector) FROM cifar_image_vector_table LIMIT {ROW_COUNT};
""")
rows = cur.fetchall()
print(f"Fetched {len(rows)} rows")

# Convert to numpy array (N, 150528) then to torch tensors (N, 1, 3, 224, 224)
all_data = np.array([r[0] for r in rows], dtype=np.float32)
all_tensors = torch.from_numpy(all_data).view(-1, 1, 3, 224, 224)
print(f"All tensors shape: {all_tensors.shape}  (N x 1 x 3 x 224 x 224)")
print(f"Data range: [{all_tensors.min().item():.4f}, {all_tensors.max().item():.4f}]")

# ── Batched inference ─────────────────────────────────────────────────────────
print(f"\nRunning batched inference (batch_size={BATCH_SIZE}, device={DEVICE})...")

# Warmup: run one batch to prime GPU kernels
warmup_batch = all_tensors[:BATCH_SIZE].to(DEVICE)
with torch.no_grad():
    _ = model(warmup_batch)
torch.cuda.synchronize(DEVICE)
print("Warmup complete")

all_outputs = []
total_inference_time = 0.0
num_batches = 0
n_samples = len(all_tensors)

for start in range(0, n_samples, BATCH_SIZE):
    batch = all_tensors[start:start + BATCH_SIZE]
    batch_tensor = batch.to(DEVICE)

    torch.cuda.synchronize(DEVICE)
    t0 = time.perf_counter()

    with torch.no_grad():
        output = model(batch_tensor)

    torch.cuda.synchronize(DEVICE)
    t1 = time.perf_counter()

    batch_time = t1 - t0
    total_inference_time += batch_time
    all_outputs.append(output)
    num_batches += 1
    print(f"  Batch {num_batches:>2d}/{(n_samples+BATCH_SIZE-1)//BATCH_SIZE:<2d}: "
          f"input={list(batch_tensor.shape)}, output={list(output.shape)}, "
          f"time={batch_time*1000:.2f}ms")

# ── Aggregate results ─────────────────────────────────────────────────────────
all_outputs = torch.cat(all_outputs, dim=0)

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Model:           googlenet_cifar10.pt (TorchScript)")
print(f"Device:          {DEVICE}")
print(f"Total samples:   {n_samples}")
print(f"Batch size:      {BATCH_SIZE}")
print(f"Num batches:     {num_batches}")
print(f"Output shape:    {list(all_outputs.shape)}")
print(f"Total time:      {total_inference_time*1000:.2f}ms ({total_inference_time:.4f}s)")
print(f"Avg per batch:   {total_inference_time/num_batches*1000:.2f}ms")
print(f"Avg per sample:  {total_inference_time/n_samples*1000:.2f}ms")
print(f"Samples/sec:     {n_samples/total_inference_time:.1f}")

cur.close()
conn.close()
