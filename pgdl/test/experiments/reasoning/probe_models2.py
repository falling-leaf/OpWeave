import time
import numpy as np
import torch
import subprocess
import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    user="why",
    password="123456",
    dbname="postgres"
)
cur = conn.cursor()

# Load 20 rows from musique_vector_test
cur.execute("SELECT get_mvec_data(reasoning_vec) FROM musique_vector_test ORDER BY id LIMIT 20")
rows_test = cur.fetchall()
input_vectors = []
for row in rows_test:
    arr = np.array(row[0], dtype=np.float32)
    arr = arr.reshape(4, 128)
    input_vectors.append(arr)

print(f"Loaded {len(input_vectors)} vectors from musique_vector_test, each shape: {input_vectors[0].shape}")

# Load 40 rows from musique_vector_step2
cur.execute("SELECT get_mvec_data(text_vec) FROM musique_vector_step2 ORDER BY id LIMIT 40")
rows_step2 = cur.fetchall()
step2_vectors = []
for row in rows_step2:
    arr = np.array(row[0], dtype=np.float32)
    step2_vectors.append(arr)

print(f"Loaded {len(step2_vectors)} vectors from musique_vector_step2, each shape: {step2_vectors[0].shape}")

# Load models on CPU
cross_encoder = torch.load("/home/why/dbagent/pgdl/test/morphingdb_test/models/cross_encoder.pt", map_location="cpu", weights_only=False)
deberta_reader = torch.load("/home/why/dbagent/pgdl/test/morphingdb_test/models/deberta_reader.pt", map_location="cpu", weights_only=False)
cross_encoder.eval()
deberta_reader.eval()

print(f"cross_encoder type: {type(cross_encoder)}")
print(f"deberta_reader type: {type(deberta_reader)}")

# Test deberta_reader with (1, 2, 256) shape
sv = step2_vectors[0]
print(f"\n--- Testing deberta_reader with (1, 2, 256) ---")
print(f"Raw step2[0] values (first 10): {sv[:10]}")
print(f"Raw step2[0] values (256-266): {sv[256:266]}")

try:
    x = torch.from_numpy(sv.reshape(1, 2, 256).astype(np.int64))
    print(f"Reshaped tensor shape: {x.shape}")
    with torch.no_grad():
        out = deberta_reader(x)
    print(f"Output type: {type(out)}")
    if isinstance(out, torch.Tensor):
        print(f"Output shape: {out.shape}")
        print(f"Output values: {out}")
    elif isinstance(out, (list, tuple)):
        print(f"Output is tuple of length {len(out)}")
        for i, item in enumerate(out):
            if isinstance(item, torch.Tensor):
                print(f"  [{i}] shape: {item.shape}, values: {item}")
except Exception as e:
    print(f"FAILED: {e}")

cur.close()
conn.close()
