import time
import numpy as np
import torch
import sys
import subprocess
import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(host="localhost", port=5432, user="why", password="123456", dbname="postgres")
cur = conn.cursor()

# Load 5 rows from musique_vector_test
cur.execute("SELECT get_mvec_data(reasoning_vec) FROM musique_vector_test ORDER BY id LIMIT 5")
test_rows = cur.fetchall()

# Load 10 rows from musique_vector_step2
cur.execute("SELECT get_mvec_data(text_vec) FROM musique_vector_step2 ORDER BY id LIMIT 10")
step2_rows = cur.fetchall()

cur.close()
conn.close()

# Reshape to (N, 4, 128)
test_data = []
for row in test_rows:
    arr = np.array(row[0], dtype=np.int32).reshape(4, 128)
    test_data.append(arr)

step2_data = []
for row in step2_rows:
    arr = np.array(row[0], dtype=np.int32).reshape(4, 128)
    step2_data.append(arr)

# Load models on CPU
cross_encoder = torch.jit.load("/home/why/dbagent/pgdl/test/morphingdb_test/models/cross_encoder.pt", map_location="cpu")
deberta_reader = torch.jit.load("/home/why/dbagent/pgdl/test/morphingdb_test/models/deberta_reader.pt", map_location="cpu")

# Print start time
print(f"EXEC_START_MS={time.perf_counter() * 1000.0:.2f}")

# Stage 1: score each of the 5 input vectors (using first 2 rows) with cross_encoder
print("\n=== Stage 1: Cross Encoder Scoring ===")
cross_scores = []
for i in range(5):
    tensor = torch.tensor(test_data[i:i+1], dtype=torch.float32)  # (1, 4, 128)
    with torch.no_grad():
        score = cross_encoder(tensor)
    cross_scores.append(score.item())
    print(f"  Sample {i}: score = {score.item():.4f}")

# Stage 2: for each sample, run deberta_reader on its 2 step2 vectors
print("\n=== Stage 2: DeBERTa Reader ===")
all_token_ids = []
for sample_idx in range(5):
    # Use 2 step2 vectors per sample (step2 indices: sample_idx*2 and sample_idx*2+1)
    s2_idx_a = sample_idx * 2
    s2_idx_b = sample_idx * 2 + 1
    if s2_idx_b >= len(step2_data):
        print(f"  Sample {sample_idx}: not enough step2 data (only have {len(step2_data)} vectors)")
        continue
    
    # Run deberta_reader on both vectors
    for s2_idx in [s2_idx_a, s2_idx_b]:
        tensor = torch.tensor(step2_data[s2_idx:s2_idx+1], dtype=torch.float32)  # (1, 4, 128)
        with torch.no_grad():
            output = deberta_reader(tensor)  # (1, 2, 128) - [start_logits, end_logits]
        
        # Extract input_ids from step2 data (row 0 of the 4x128 tensor)
        input_ids = step2_data[s2_idx][0]  # (128,)
        
        # Get start/end positions via argmax
        start_logits = output[0, 0, :]  # (128,)
        end_logits = output[0, 1, :]    # (128,)
        
        start_pos = torch.argmax(start_logits).item()
        end_pos = torch.argmax(end_logits).item()
        
        # Clamp end_pos to be >= start_pos
        if end_pos < start_pos:
            end_pos = start_pos
        
        # Extract token IDs between start and end positions
        token_ids = input_ids[start_pos:end_pos + 1].tolist()
        all_token_ids.append((sample_idx, s2_idx, token_ids))
        
        print(f"  Sample {sample_idx}, step2[{s2_idx}]: start={start_pos}, end={end_pos}, ids={token_ids[:10]}... ({len(token_ids)} total)")

# Stage 3: decode token IDs using subprocess
print("\n=== Stage 3: Decoding ===")
decoded_count = 0
for sample_idx, s2_idx, token_ids in all_token_ids:
    ids_string = ",".join(str(x) for x in token_ids)
    
    result = subprocess.run(
        ["uv", "run", "/home/why/dbagent/pgdl/test/tools/reader_decoder.py", "--ids", ids_string],
        cwd="/home/why/dbagent/pgdl/test/tools",
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode == 0:
        decoded_count += 1
        output_lines = result.stdout.strip().split('\n')
        # Filter out the "Execution time" line
        json_output = [l for l in output_lines if not l.startswith("Execution time:")]
        print(f"  Sample {sample_idx}, step2[{s2_idx}]: {''.join(json_output[:2])}")
    else:
        print(f"  Sample {sample_idx}, step2[{s2_idx}]: DECODE FAILED (rc={result.returncode})")
        if result.stderr:
            print(f"    stderr: {result.stderr[:200]}")

# Print end time
print(f"\nEXEC_END_MS={time.perf_counter() * 1000.0:.2f}")
print(f"\nDecoded answers: {decoded_count}/{len(all_token_ids)}")
