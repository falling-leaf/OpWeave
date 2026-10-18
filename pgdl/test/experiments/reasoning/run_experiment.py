import time
import numpy as np
import torch
import sys
import subprocess
import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(host='localhost', port=5432, user='why', password='123456', dbname='postgres')
cur = conn.cursor()

# Load 1 row from musique_vector_test
cur.execute('SELECT get_mvec_data(reasoning_vec) FROM musique_vector_test ORDER BY id LIMIT 1')
test_rows = cur.fetchall()
reasoning_vec = np.array(test_rows[0][0], dtype=np.float64).reshape(4, 128)

# Load 2 rows from musique_vector_step2
cur.execute('SELECT get_mvec_data(text_vec) FROM musique_vector_step2 ORDER BY id LIMIT 2')
step2_rows = cur.fetchall()
step2_vecs = [np.array(r[0], dtype=np.float64).reshape(4, 128) for r in step2_rows]

cur.close()
conn.close()

# Load models on CPU
cross_encoder = torch.jit.load('/home/why/dbagent/pgdl/test/morphingdb_test/models/cross_encoder.pt', map_location='cpu')
cross_encoder.eval()

deberta_reader = torch.jit.load('/home/why/dbagent/pgdl/test/morphingdb_test/models/deberta_reader.pt', map_location='cpu')
deberta_reader.eval()

print(f"EXEC_START_MS={time.perf_counter()*1000.0:.2f}")

# Stage 1: score each input vector with cross_encoder
# reasoning_vec is (4, 128) -> interpret as (2, 2, 128): rows 0,2=input_ids, rows 1,3=mask
# For each of the 1 input vectors, pair with first 2 step2 rows
reasoning_input_ids = torch.tensor(reasoning_vec[0:1, :], dtype=torch.int64)  # (1, 128)
reasoning_mask = torch.tensor(reasoning_vec[1:2, :], dtype=torch.int64)        # (1, 128)
reasoning_input = torch.stack([reasoning_input_ids, reasoning_mask], dim=1)     # (1, 2, 128)

step2_0_input_ids = torch.tensor(step2_vecs[0][0:1, :], dtype=torch.int64)
step2_0_mask = torch.tensor(step2_vecs[0][1:2, :], dtype=torch.int64)
step2_0_input = torch.stack([step2_0_input_ids, step2_0_mask], dim=1)

step2_1_input_ids = torch.tensor(step2_vecs[1][0:1, :], dtype=torch.int64)
step2_1_mask = torch.tensor(step2_vecs[1][1:2, :], dtype=torch.int64)
step2_1_input = torch.stack([step2_1_input_ids, step2_1_mask], dim=1)

# Score reasoning_vec against each step2 vector
with torch.no_grad():
    score_0 = cross_encoder(reasoning_input)
    score_1 = cross_encoder(step2_0_input)
    score_2 = cross_encoder(step2_1_input)

print(f"Stage 1 scores: reasoning={score_0.tolist()}, step2_0={score_1.tolist()}, step2_2={score_2.tolist()}")

# Stage 2: for each sample, run deberta_reader on its 2 step2 vectors
# Each step2 vector: (4, 128) -> (2, 2, 128)
decoded_answers = 0

for vec_idx, step2_vec in enumerate(step2_vecs):
    s2_input_ids = torch.tensor(step2_vec[0:1, :], dtype=torch.int64)
    s2_mask = torch.tensor(step2_vec[1:2, :], dtype=torch.int64)
    s2_input = torch.stack([s2_input_ids, s2_mask], dim=1)
    
    with torch.no_grad():
        reader_output = deberta_reader(s2_input)
    
    # reader_output is (batch, 2, seq_len) - start_logits and end_logits
    start_logits = reader_output[:, 0, :]
    end_logits = reader_output[:, 1, :]
    
    # Get token IDs by argmax
    start_ids = torch.argmax(start_logits, dim=-1).tolist()
    end_ids = torch.argmax(end_logits, dim=-1).tolist()
    
    print(f"Stage 2 step2_vec[{vec_idx}]: start_ids={start_ids}, end_ids={end_ids}")
    
    # Stage 3: decode token IDs via subprocess
    # Format: "start_id,end_id" as ids_string
    ids_string = f"{start_ids[0]},{end_ids[0]}"
    result = subprocess.run(
        ['uv', 'run', '/home/why/dbagent/pgdl/test/tools/reader_decoder.py', '--ids', ids_string],
        cwd='/home/why/dbagent/pgdl/test/tools',
        capture_output=True, text=True, timeout=120
    )
    print(f"Stage 3 decoded[{vec_idx}]: {result.stdout.strip()}")
    decoded_answers += 1

print(f"EXEC_END_MS={time.perf_counter()*1000.0:.2f}")
print(f"Decoded answers: {decoded_answers}")
