import time
import numpy as np
import torch
import sys
import subprocess
import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(host='localhost', port=5432, user='why', password='123456', dbname='postgres')
cur = conn.cursor()

# Load 20 rows from reasoning_vector_test
cur.execute('SELECT get_mvec_data(reasoning_vec) FROM reasoning_vector_test ORDER BY id LIMIT 20')
test_rows = cur.fetchall()
test_vectors = []
for row in test_rows:
    arr = np.array(row[0], dtype=np.float32).reshape(4, 128)
    test_vectors.append(arr)
print(f"Loaded {len(test_vectors)} test vectors, shape each: {test_vectors[0].shape}")

# Load 40 rows from reasoning_vector_step2
cur.execute('SELECT get_mvec_data(text_vec) FROM reasoning_vector_step2 ORDER BY id LIMIT 40')
step2_rows = cur.fetchall()
step2_vectors = []
for row in step2_rows:
    arr = np.array(row[0], dtype=np.float32).reshape(4, 128)
    step2_vectors.append(arr)
print(f"Loaded {len(step2_vectors)} step2 vectors, shape each: {step2_vectors[0].shape}")

conn.close()

# Load models on CPU
print("Loading cross_encoder...")
cross_encoder = torch.jit.load('/home/why/dbagent/pgdl/test/morphingdb_test/models/cross_encoder.pt', map_location='cpu')
cross_encoder.eval()

print("Loading deberta_reader...")
deberta_reader = torch.jit.load('/home/why/dbagent/pgdl/test/morphingdb_test/models/deberta_reader.pt', map_location='cpu')
deberta_reader.eval()

print(f"EXEC_START_MS={time.perf_counter()*1000.0}")

# Stage 1: Score each of the 20 input vectors with cross_encoder (using first 2 rows)
print("\n--- Stage 1: Cross-encoder scoring ---")
cross_scores = []
for i, vec in enumerate(test_vectors):
    input_ids = vec[0, :].astype(np.int64)      # Row 0: token IDs
    attention_mask = vec[1, :].astype(np.int64)  # Row 1: attention mask
    # Shape: (1, 2, 128)
    inp = torch.tensor(np.stack([input_ids, attention_mask]), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        score = cross_encoder(inp)
    cross_scores.append(score.item())
    if i < 3:
        print(f"  Sample {i}: score={score.item():.4f}")
print(f"  Scored {len(cross_scores)} samples")

# Stage 2: Run deberta_reader on each sample's 2 step2 vectors
print("\n--- Stage 2: DeBERTa reader inference ---")
reader_outputs = []
for i in range(20):
    # Each sample gets 2 step2 vectors
    v0 = step2_vectors[i * 2]
    v1 = step2_vectors[i * 2 + 1]
    
    # Combine into (2, 2, 128): each of 2 samples has (input_ids, attention_mask)
    inp_combined = torch.cat([
        torch.tensor(v0[:2, :], dtype=torch.float32).unsqueeze(0),  # (1, 2, 128)
        torch.tensor(v1[:2, :], dtype=torch.float32).unsqueeze(0),  # (1, 2, 128)
    ], dim=0)  # (2, 2, 128)
    
    inp = inp_combined
    
    with torch.no_grad():
        start_logits, end_logits = deberta_reader(inp)
    
    # Extract token IDs for each of the 2 vectors
    for j in range(2):
        input_ids_j = inp[j, 0, :].long()  # Extract input_ids for this vector
        start = torch.argmax(start_logits[j]).item()
        end = torch.argmax(end_logits[j]).item()
        if end < start:
            end = start
        token_ids = input_ids_j[start:end+1].tolist()
        reader_outputs.append(token_ids)
    
    if i < 3:
        print(f"  Sample {i}: start={start_logits.argmax(dim=1).tolist()}, end={end_logits.argmax(dim=1).tolist()}")

print(f"  Generated {len(reader_outputs)} token ID sequences")

# Stage 3: Decode token IDs using reader_decoder.py
print("\n--- Stage 3: Decoding ---")
decoded_count = 0
for i, token_ids in enumerate(reader_outputs):
    if not token_ids:
        print(f"  Segment {i}: empty (skipped)")
        continue
    ids_str = ",".join(str(x) for x in token_ids)
    try:
        result = subprocess.run(
            ["uv", "run", "/home/why/dbagent/pgdl/test/tools/reader_decoder.py", "--ids", ids_str],
            cwd="/home/why/dbagent/pgdl/test/tools",
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            decoded_count += 1
            # Print first 80 chars of decoded text
            import json
            try:
                data = json.loads(result.stdout)
                text = data.get("paragraph_1", "")[:80]
                print(f"  Segment {i}: {text}")
            except:
                print(f"  Segment {i}: {result.stdout[:80]}")
        else:
            print(f"  Segment {i}: FAILED (rc={result.returncode})")
    except subprocess.TimeoutExpired:
        print(f"  Segment {i}: TIMEOUT")
    except Exception as e:
        print(f"  Segment {i}: ERROR {e}")

print(f"EXEC_END_MS={time.perf_counter()*1000.0}")
print(f"\nDecoded answers: {decoded_count} out of {len(reader_outputs)}")
