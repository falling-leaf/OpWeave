"""
Batch inference script: Load GoogLeNet CIFAR-10 model, fetch 1000 image vectors
from PostgreSQL cifar_image_vector_table, run batched inference on GPU.
"""
import sys
import os
import time
import torch
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH = "/home/why/baseline/models/googlenet_cifar10.pt"
DB_CONFIG = {
    "host": "localhost",
    "user": "why",
    "password": "123456",
    "dbname": "postgres",
}
TABLE_NAME = "cifar_image_vector_table"
BATCH_SIZE = 32
DEVICE = "cuda:0"


def fetch_mvec_data(conn, table, column, limit=1000):
    """Fetch mvec data and shape from PostgreSQL.

    mvec columns expose get_mvec_data() -> float[] and get_mvec_shape() -> float[].
    """
    cur = conn.cursor()
    cur.execute(f"""
        SELECT get_mvec_data({column}), get_mvec_shape({column})
        FROM {table}
        LIMIT {limit}
    """)
    rows = cur.fetchall()
    cur.close()
    return rows


def main():
    print(f"Loading model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    model = torch.jit.load(MODEL_PATH).to(DEVICE)
    model.eval()
    print(f"Model loaded successfully (type: {type(model)})")

    # Probe input/output shape with a dummy tensor
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        dummy_out = model(dummy)
    print(f"Model input shape:  [{1, 3, 224, 224}]")
    print(f"Model output shape: {list(dummy_out.shape)}")

    # ── Fetch data from PostgreSQL ─────────────────────────────────────────────
    print(f"\nFetching image vectors from {TABLE_NAME} ...")
    conn = psycopg2.connect(**DB_CONFIG)
    rows = fetch_mvec_data(conn, TABLE_NAME, "image_vector", limit=1000)
    conn.close()
    print(f"Fetched {len(rows)} image vectors")

    if len(rows) == 0:
        print("ERROR: No data found. Check table name and column name.")
        return

    # Inspect first row to understand shape
    first_data = rows[0][0]
    first_shape = rows[0][1]
    print(f"First mvec shape from DB: {first_shape}")
    print(f"First mvec length (elements): {len(first_data)}")

    # ── Convert to tensors ─────────────────────────────────────────────────────
    print("\nConverting mvec data to GPU tensors ...")
    # Each mvec is a flat float list. Reconstruct based on expected shape [1, 3, 224, 224]
    # or whatever the DB reports. We'll use the actual shape from the first row.
    expected_shape = tuple(int(float(s)) for s in first_shape)
    print(f"Expected tensor shape per sample: {expected_shape}")

    tensors = []
    for i, (data, _) in enumerate(rows):
        arr = torch.tensor(data, dtype=torch.float32, device=DEVICE)
        tensor = arr.reshape(expected_shape)
        tensors.append(tensor)

    all_data = torch.cat(tensors, dim=0)
    total = all_data.shape[0]
    print(f"Total tensor shape: {list(all_data.shape)}")
    print(f"Total samples: {total}")

    # ── Batched inference ──────────────────────────────────────────────────────
    print(f"\nRunning batched inference (batch_size={BATCH_SIZE}) ...")
    all_outputs = []
    inference_start = time.perf_counter()

    with torch.no_grad():
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch = all_data[start:end]
            output = model(batch)
            all_outputs.append(output)

    # Synchronize CUDA to ensure all GPU work is complete before stopping timer
    torch.cuda.synchronize(DEVICE)
    inference_end = time.perf_counter()

    all_outputs = torch.cat(all_outputs, dim=0)
    elapsed_ms = (inference_end - inference_start) * 1000.0

    # ── Report ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Model:              {MODEL_PATH}")
    print(f"  Device:             {DEVICE}")
    print(f"  Input samples:      {total}")
    print(f"  Input shape:        {list(all_data.shape)}")
    print(f"  Output shape:       {list(all_outputs.shape)}")
    print(f"  Output dtype:       {all_outputs.dtype}")
    print(f"  Batch size:         {BATCH_SIZE}")
    print(f"  Num batches:        {(total + BATCH_SIZE - 1) // BATCH_SIZE}")
    print(f"  Total inference:    {elapsed_ms:.2f} ms  ({elapsed_ms/1000:.4f} s)")
    print(f"  Per-sample avg:     {elapsed_ms/total:.4f} ms")
    print(f"  Samples per second: {total / (elapsed_ms/1000):.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
