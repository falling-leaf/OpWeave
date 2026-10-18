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

print(f"Loaded {len(step2_vectors)} vectors from musique_vector_step2")
print(f"  step2[0] shape: {step2_vectors[0].shape}, dtype: {step2_vectors[0].dtype}")
print(f"  step2[0] sample: {step2_vectors[0][:10]}")

# Check if step2 vectors are token IDs (integers) or embeddings
print(f"  step2[0] min: {step2_vectors[0].min()}, max: {step2_vectors[0].max()}")

# Load models on CPU
cross_encoder = torch.load("/home/why/dbagent/pgdl/test/morphingdb_test/models/cross_encoder.pt", map_location="cpu", weights_only=False)
deberta_reader = torch.load("/home/why/dbagent/pgdl/test/morphingdb_test/models/deberta_reader.pt", map_location="cpu", weights_only=False)
cross_encoder.eval()
deberta_reader.eval()

print(f"cross_encoder type: {type(cross_encoder)}")
print(f"deberta_reader type: {type(deberta_reader)}")

# Try different input shapes for deberta_reader to find what works
print("\n--- Probing deberta_reader input format ---")
sv = step2_vectors[0]
print(f"Raw step2 vector shape: {sv.shape}")

# Try as 1D tensor
try:
    x1 = torch.from_numpy(sv)
    print(f"1D tensor shape: {x1.shape}")
    with torch.no_grad():
        out1 = deberta_reader(x1)
    print(f"1D input -> output shape: {out1.shape}")
except Exception as e:
    print(f"1D input FAILED: {e}")

# Try as 2D tensor (1, N)
try:
    x2 = torch.from_numpy(sv).unsqueeze(0)
    print(f"2D tensor shape: {x2.shape}")
    with torch.no_grad():
        out2 = deberta_reader(x2)
    print(f"2D input -> output shape: {out2.shape}")
except Exception as e:
    print(f"2D input FAILED: {e}")

# Try as 3D tensor (1, 1, N)
try:
    x3 = torch.from_numpy(sv).unsqueeze(0).unsqueeze(0)
    print(f"3D tensor shape: {x3.shape}")
    with torch.no_grad():
        out3 = deberta_reader(x3)
    print(f"3D input -> output shape: {out3.shape}")
except Exception as e:
    print(f"3D input FAILED: {e}")

# Try reshaped to (4, 128) 
try:
    if sv.shape[0] == 512:
        x4 = torch.from_numpy(sv.reshape(4, 128))
        print(f"4x128 tensor shape: {x4.shape}")
        with torch.no_grad():
            out4 = deberta_reader(x4)
        print(f"4x128 input -> output shape: {out4.shape}")
except Exception as e:
    print(f"4x128 input FAILED: {e}")

# Try as long tensor (token IDs)
try:
    x5 = torch.from_numpy(sv.astype(np.int64)).unsqueeze(0)
    print(f"Long tensor shape: {x5.shape}, dtype: {x5.dtype}")
    with torch.no_grad():
        out5 = deberta_reader(x5)
    print(f"Long input -> output shape: {out5.shape}")
except Exception as e:
    print(f"Long input FAILED: {e}")

cur.close()
conn.close()
