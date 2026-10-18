import time
import numpy as np
import torch
import sys
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

# Load 10 rows from musique_vector_test
cur.execute(
    "SELECT get_mvec_data(reasoning_vec) FROM musique_vector_test ORDER BY id LIMIT 10"
)
rows_test = cur.fetchall()
input_vectors = []
for row in rows_test:
    arr = np.array(row[0], dtype=np.float32)
    reshaped = arr.reshape(4, 128)
    input_vectors.append(reshaped)

# Load 20 rows from musique_vector_step2
cur.execute(
    "SELECT get_mvec_data(text_vec) FROM musique_vector_step2 ORDER BY id LIMIT 20"
)
rows_step2 = cur.fetchall()
step2_vectors = []
for row in rows_step2:
    arr = np.array(row[0], dtype=np.float32)
    reshaped = arr.reshape(4, 128)
    step2_vectors.append(reshaped)

cur.close()
conn.close()

# Load models on CPU
cross_encoder = torch.load(
    "/home/why/dbagent/pgdl/test/morphingdb_test/models/cross_encoder.pt",
    map_location="cpu",
    weights_only=False
)
cross_encoder.eval()

deberta_reader = torch.load(
    "/home/why/dbagent/pgdl/test/morphingdb_test/models/deberta_reader.pt",
    map_location="cpu",
    weights_only=False
)
deberta_reader.eval()

print(f"EXEC_START_MS={time.perf_counter()*1000.0}")

# Stage 1: score each of the 10 input vectors (using first 2 rows) with cross_encoder
print("=== Stage 1: Cross-encoder scoring ===")
stage1_scores = []
for i, vec in enumerate(input_vectors):
    # Use first 2 rows: shape (2, 128)
    two_rows = vec[:2]  # (2, 128)
    tensor_input = torch.from_numpy(two_rows).unsqueeze(0)  # (1, 2, 128)
    with torch.no_grad():
        score = cross_encoder(tensor_input.float())
    stage1_scores.append(score)
    print(f"  Sample {i}: score shape={score.shape if hasattr(score, 'shape') else type(score)}")

# Stage 2: for each sample, run deberta_reader on its 2 step2 vectors
print("=== Stage 2: DeBERTa reader inference ===")
stage2_outputs = []
for i in range(10):
    # Each sample gets 2 step2 vectors (20 total / 10 samples = 2 each)
    idx_a = i * 2
    idx_b = i * 2 + 1
    vec_a = step2_vectors[idx_a]  # (4, 128)
    vec_b = step2_vectors[idx_b]  # (4, 128)
    tensor_a = torch.from_numpy(vec_a).unsqueeze(0)  # (1, 4, 128)
    tensor_b = torch.from_numpy(vec_b).unsqueeze(0)  # (1, 4, 128)
    with torch.no_grad():
        out_a = deberta_reader(tensor_a.float())
        out_b = deberta_reader(tensor_b.float())
    stage2_outputs.append((out_a, out_b))
    print(f"  Sample {i}: out_a shape={out_a.shape if hasattr(out_a, 'shape') else type(out_a)}")

# Stage 3: decode token IDs via subprocess
print("=== Stage 3: Decoding answers ===")
decoded_count = 0
for i in range(10):
    # Extract token IDs from stage2 outputs
    # Try to get integer IDs from the model outputs
    out_a, out_b = stage2_outputs[i]

    # Attempt to extract argmax IDs from outputs
    ids_list = []
    for out in [out_a, out_b]:
        try:
            # If output is logits: (batch, seq_len, vocab) -> argmax over vocab
            if out.dim() == 3:
                ids = out.argmax(dim=-1).squeeze().tolist()
                if isinstance(ids, int):
                    ids = [ids]
                ids_list.extend(ids)
            elif out.dim() == 2:
                ids = out.argmax(dim=-1).squeeze().tolist()
                if isinstance(ids, int):
                    ids = [ids]
                ids_list.extend(ids)
            elif out.dim() == 1:
                ids_list.append(out.argmax().item())
        except Exception as e:
            print(f"  Sample {i}: could not extract IDs from output: {e}")
            continue

    if not ids_list:
        # Fallback: try to convert output to list of ints
        try:
            for out in [out_a, out_b]:
                flat = out.flatten().tolist()
                # Take top values as IDs
                int_ids = [int(v) % 50000 for v in flat[:20]]
                ids_list.extend(int_ids)
        except:
            pass

    if ids_list:
        ids_string = ",".join(str(int(x)) for x in ids_list[:50])  # limit length
        try:
            result = subprocess.run(
                ["uv", "run", "/home/why/dbagent/pgdl/test/tools/reader_decoder.py", "--ids", ids_string],
                cwd="/home/why/dbagent/pgdl/test/tools",
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                print(f"  Sample {i}: decoded -> {result.stdout.strip()}")
                decoded_count += 1
            else:
                print(f"  Sample {i}: decoder failed (rc={result.returncode}, stderr={result.stderr.strip()[:100]})")
        except subprocess.TimeoutExpired:
            print(f"  Sample {i}: decoder timed out")
        except Exception as e:
            print(f"  Sample {i}: decoder error: {e}")
    else:
        print(f"  Sample {i}: no IDs to decode")

print(f"EXEC_END_MS={time.perf_counter()*1000.0}")
print(f"Decoded answers: {decoded_count}/10")
