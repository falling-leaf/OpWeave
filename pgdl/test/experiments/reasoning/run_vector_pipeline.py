import time
import numpy as np
import torch
import sys
import subprocess
import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(host='localhost', port=5432, user='why', password='123456', dbname='postgres')
cur = conn.cursor()

# Load 10 rows from musique_vector_test
cur.execute("SELECT get_mvec_data(reasoning_vec) FROM musique_vector_test ORDER BY id LIMIT 10")
test_rows = cur.fetchall()
test_data = []
for row in test_rows:
    arr = np.array(row[0], dtype=np.float32).reshape(4, 128)
    test_data.append(arr)

# Load 20 rows from musique_vector_step2
cur.execute("SELECT get_mvec_data(text_vec) FROM musique_vector_step2 ORDER BY id LIMIT 20")
step2_rows = cur.fetchall()
step2_data = []
for row in step2_rows:
    arr = np.array(row[0], dtype=np.float32).reshape(4, 128)
    step2_data.append(arr)

cur.close()
conn.close()

print(f"Loaded {len(test_data)} test vectors, {len(step2_data)} step2 vectors")

# Load models on CPU
cross_encoder = torch.jit.load(
    '/home/why/dbagent/pgdl/test/morphingdb_test/models/cross_encoder.pt',
    map_location='cpu'
)
deberta_reader = torch.jit.load(
    '/home/why/dbagent/pgdl/test/morphingdb_test/models/deberta_reader.pt',
    map_location='cpu'
)
cross_encoder.eval()
deberta_reader.eval()

print(f"EXEC_START_MS={time.perf_counter() * 1000.0:.2f}")

# Stage 1: score each of the 10 input vectors (using first 2 rows) with cross_encoder
# Each test_data[i] is (4,128); first 2 rows -> (2,128) -> input_ids + mask
# Stack all 10 samples: (10, 2, 128)
stage1_inputs_list = []
for i in range(10):
    first_two = test_data[i][:2]  # (2, 128)
    stage1_inputs_list.append(first_two)
stage1_batch = torch.from_numpy(np.stack(stage1_inputs_list))  # (10, 2, 128)

with torch.no_grad():
    stage1_scores = cross_encoder(stage1_batch)

print(f"Stage 1 output shape: {stage1_scores.shape}")
print(f"Stage 1 scores: {stage1_scores.flatten()[:6].tolist()}")

# Stage 2: for each sample, run deberta_reader on its 2 step2 vectors
# Sample i uses step2_data[2*i] and step2_data[2*i+1]
# Take first 2 rows of each (4,128) step2 vector
stage2_outputs = []
for i in range(10):
    # Each sample gets 2 step2 vectors
    vec_a = step2_data[2 * i][:2]   # first 2 rows of (4,128)
    vec_b = step2_data[2 * i + 1][:2]
    
    # Run each through deberta_reader (batch of 2)
    batch2 = torch.from_numpy(np.stack([vec_a, vec_b]))  # (2, 2, 128)
    
    with torch.no_grad():
        out = deberta_reader(batch2)  # (2, 2, 128) -> [start_logits, end_logits] stacked
        stage2_outputs.append(out)
    
    print(f"  Sample {i} deberta output shape: {out.shape}")

# Stage 3: decode token IDs by calling subprocess
# From deberta output: out[i, 0, :] = start_logits, out[i, 1, :] = end_logits
# Extract answer spans from input_ids
decoded_answers = []
for i in range(10):
    # Use the first step2 vector's output (index 0 of the batch of 2)
    sample_out = stage2_outputs[i][0]  # (2, 128)
    start_logits = sample_out[0]  # (128,)
    end_logits = sample_out[1]    # (128,)
    
    # Argmax to get positions
    start_pos = int(torch.argmax(start_logits).item())
    end_pos = int(torch.argmax(end_logits).item())
    
    # Get input_ids from the step2 data (first row = input_ids, second = mask)
    input_ids_1d = step2_data[2 * i][0]  # first row of (4,128) = input_ids
    token_ids = input_ids_1d.astype(int).tolist()
    
    # Extract answer span
    answer_ids = token_ids[start_pos:end_pos + 1] if start_pos <= end_pos else token_ids[start_pos:start_pos + 10]
    
    ids_string = ",".join(str(x) for x in answer_ids)
    
    if ids_string:
        try:
            result = subprocess.run(
                ['uv', 'run', '/home/why/dbagent/pgdl/test/tools/reader_decoder.py', '--ids', ids_string],
                cwd='/home/why/dbagent/pgdl/test/tools',
                capture_output=True,
                text=True,
                timeout=60
            )
            decoded_answers.append(result.stdout.strip())
            print(f"  Sample {i} decoded: {result.stdout.strip()[:120]}")
        except subprocess.TimeoutExpired:
            print(f"  Sample {i} decode timeout")
        except Exception as e:
            print(f"  Sample {i} decode error: {e}")
    else:
        print(f"  Sample {i} empty ids_string")

print(f"EXEC_END_MS={time.perf_counter() * 1000.0:.2f}")
print(f"Answers decoded: {len(decoded_answers)}")
