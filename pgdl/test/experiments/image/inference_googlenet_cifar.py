#!/usr/bin/env python3
"""
Batch inference on 1000 CIFAR image vectors from PostgreSQL using GoogLeNet.
- Loads model via torch.jit.load on GPU (cuda:0)
- Fetches 1000 mvec vectors from cifar_image_vector_table
- Runs batched inference with batch_size=32
- Reports output shape and precise inference time
"""
import time
import numpy as np
import torch
import psycopg2

MODEL_PATH = "/home/why/baseline/models/googlenet_cifar10.pt"
DB_CONFIG = {
    "dbname": "postgres",
    "host": "localhost",
    "port": "5432",
    "user": "why",
    "password": "123456",
}
BATCH_SIZE = 32
NUM_IMAGES = 1000
# 1x3x224x224 = 150528 elements per image
EXPECTED_ELEMENTS = 1 * 3 * 224 * 224

def fetch_vectors():
    """Fetch NUM_IMAGES mvec vectors from PostgreSQL."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "SELECT get_mvec_data(image_vector) FROM cifar_image_vector_table ORDER BY id LIMIT %s",
        (NUM_IMAGES,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def main():
    device = torch.device("cuda:0")

    # --- Load model ---
    print(f"Loading model from {MODEL_PATH} ...")
    t_load_start = time.perf_counter()
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()
    t_load = time.perf_counter() - t_load_start
    print(f"  Model loaded in {t_load*1000:.1f} ms")

    # --- Fetch data from PostgreSQL ---
    print(f"Fetching {NUM_IMAGES} image vectors from cifar_image_vector_table ...")
    t_fetch_start = time.perf_counter()
    rows = fetch_vectors()
    t_fetch = time.perf_counter() - t_fetch_start
    print(f"  Fetched {len(rows)} vectors in {t_fetch*1000:.1f} ms")

    # --- Verify and convert to tensor ---
    # Each row[0] is a list of floats from get_mvec_data()
    # Expected shape per image: [1, 3, 224, 224] = 150528 elements
    first_vec = rows[0][0]
    vec_len = len(first_vec)
    print(f"  First vector length: {vec_len} (expected {EXPECTED_ELEMENTS})")

    if vec_len != EXPECTED_ELEMENTS:
        print(f"  Note: vector length {vec_len} differs from expected {EXPECTED_ELEMENTS}")

    print("Converting vectors to PyTorch tensor ...")
    t_convert_start = time.perf_counter()
    all_np = np.zeros((NUM_IMAGES, vec_len), dtype=np.float32)
    for i, row in enumerate(rows):
        all_np[i] = np.array(row[0], dtype=np.float32)
    # Reshape based on actual vector length
    if vec_len == 150528:
        all_np = all_np.reshape(NUM_IMAGES, 3, 224, 224)
    elif vec_len == 3 * 32 * 32:  # raw CIFAR-10
        all_np = all_np.reshape(NUM_IMAGES, 3, 32, 32)
    else:
        # Keep as (N, vec_len) and let the model handle it
        pass
    
    input_tensor = torch.from_numpy(all_np).to(device, non_blocking=True)
    t_convert = time.perf_counter() - t_convert_start
    print(f"  Converted to tensor shape {input_tensor.shape} in {t_convert*1000:.1f} ms")

    # --- Batched inference ---
    print(f"Running batched inference (batch_size={BATCH_SIZE}) ...")
    torch.cuda.synchronize(device)
    t_infer_start = time.perf_counter()
    
    outputs = []
    num_batches = (NUM_IMAGES + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(num_batches):
        start_idx = b * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, NUM_IMAGES)
        batch = input_tensor[start_idx:end_idx]
        with torch.no_grad():
            output = model(batch)
        outputs.append(output)
    
    torch.cuda.synchronize(device)
    t_infer = time.perf_counter() - t_infer_start
    
    # Concatenate all batch outputs
    all_outputs = torch.cat(outputs, dim=0)
    
    print(f"\n=== Results ===")
    print(f"  Total images:       {NUM_IMAGES}")
    print(f"  Batch size:         {BATCH_SIZE}")
    print(f"  Number of batches:  {num_batches}")
    print(f"  Output shape:       {all_outputs.shape}")
    print(f"  Inference time:     {t_infer*1000:.2f} ms total")
    print(f"  Per image:          {t_infer/NUM_IMAGES*1000:.4f} ms")
    print(f"  Throughput:         {NUM_IMAGES/t_infer:.2f} images/sec")
    print(f"  GPU device:         {device}")
    
    # Sample output values
    print(f"\n  Sample output (first image, first 10 logits): {all_outputs[0, :10].cpu().tolist()}")
    print(f"  Predicted class (first 5): {all_outputs[:5].argmax(dim=1).cpu().tolist()}")

if __name__ == "__main__":
    main()
