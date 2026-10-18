import time
import numpy as np
import torch
import sys
import subprocess
import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(host="localhost", port=5432, user="why", password="123456", dbname="postgres")
cur = conn.cursor()

# Load 1 row from reasoning_vector_test
cur.execute("SELECT get_mvec_data(reasoning_vec) FROM reasoning_vector_test ORDER BY id LIMIT 1")
row1 = cur.fetchone()[0]
vec1 = np.array(row1, dtype=np.float32).reshape(4, 128)

# Load 2 rows from reasoning_vector_step2
cur.execute("SELECT get_mvec_data(text_vec) FROM reasoning_vector_step2 ORDER BY id LIMIT 2")
rows2 = cur.fetchall()
vec2_a = np.array(rows2[0][0], dtype=np.float32).reshape(4, 128)
vec2_b = np.array(rows2[1][0], dtype=np.float32).reshape(4, 128)

cur.close()
conn.close()

# Load models on CPU
cross_encoder = torch.jit.load("/home/why/dbagent/pgdl/test/morphingdb_test/models/cross_encoder.pt", map_location="cpu")
cross_encoder.eval()
deberta_reader = torch.jit.load("/home/why/dbagent/pgdl/test/morphingdb_test/models/deberta_reader.pt", map_location="cpu")
deberta_reader.eval()

print(f"EXEC_START_MS={time.perf_counter()*1000.0}")

# Stage 1: score input vector with cross_encoder
# Reshaped vec1 is (4,128): row0=input_ids, row1=mask
# Cross encoder expects (batch, 2, seq_len) where dim1: 0=input_ids, 1=mask
ce_input = torch.from_numpy(np.stack([vec1[0], vec1[1]], axis=0)).unsqueeze(0).to(torch.long)
ce_input = ce_input.to(torch.int64)
with torch.no_grad():
    ce_output = cross_encoder(ce_input)
print(f"Stage 1 cross_encoder output shape: {ce_output.shape}")
print(f"Stage 1 cross_encoder scores: {ce_output}")

# Stage 2: run deberta_reader on each step2 vector
# DeBERTa reader expects (batch, 2, seq_len) and returns stacked (start_logits, end_logits)
reader_inputs = []
for label, vec in [("sample_A", vec2_a), ("sample_B", vec2_b)]:
    dr_input = torch.from_numpy(np.stack([vec[0], vec[1]], axis=0)).unsqueeze(0).to(torch.long)
    with torch.no_grad():
        dr_output = deberta_reader(dr_input)
    print(f"Stage 2 {label} deberta_reader output shape: {dr_output.shape}")
    reader_inputs.append((label, dr_output))

# Stage 3: decode token IDs from reader outputs
# dr_output shape: (2, batch=1, seq_len) -> row0=start_logits, row1=end_logits
# Extract top-1 token IDs for each sample
decoded_count = 0
for label, dr_output in reader_inputs:
    start_logits = dr_output[0, 0]  # (seq_len,)
    end_logits = dr_output[0, 1]    # (seq_len,)
    start_id = torch.argmax(start_logits).item()
    end_id = torch.argmax(end_logits).item()
    ids_string = f"{start_id},{end_id}"
    print(f"Stage 3 {label} token IDs: {ids_string}")
    result = subprocess.run(
        ["uv", "run", "/home/why/dbagent/pgdl/test/tools/reader_decoder.py", "--ids", ids_string],
        cwd="/home/why/dbagent/pgdl/test/tools",
        capture_output=True,
        text=True,
        timeout=60
    )
    print(f"Stage 3 {label} decoded output: {result.stdout.strip()}")
    if result.returncode == 0:
        decoded_count += 1

print(f"EXEC_END_MS={time.perf_counter()*1000.0}")
print(f"Decoded answers: {decoded_count}")
