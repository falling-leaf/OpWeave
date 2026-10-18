import time
import numpy as np
import torch
import sys
import subprocess
import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(host='localhost', port=5432, user='why', password='123456', dbname='postgres')
cur = conn.cursor()

# Load 5 rows from musique_vector_test
cur.execute('SELECT get_mvec_data(reasoning_vec) FROM musique_vector_test ORDER BY id LIMIT 5')
test_rows = cur.fetchall()
test_vecs = [np.array(row[0], dtype=np.float32).reshape(4, 128) for row in test_rows]
print(f"Loaded {len(test_vecs)} test vectors, each shape (4, 128)")

# Load 10 rows from musique_vector_step2
cur.execute('SELECT get_mvec_data(text_vec) FROM musique_vector_step2 ORDER BY id LIMIT 10')
step2_rows = cur.fetchall()
step2_vecs = [np.array(row[0], dtype=np.float32).reshape(4, 128) for row in step2_rows]
print(f"Loaded {len(step2_vecs)} step2 vectors, each shape (4, 128)")

cur.close()
conn.close()

# Load models on CPU
cross_encoder = torch.jit.load('/home/why/dbagent/pgdl/test/morphingdb_test/models/cross_encoder.pt', map_location='cpu')
cross_encoder.eval()
print(f"Loaded cross_encoder on CPU")

deberta_reader = torch.jit.load('/home/why/dbagent/pgdl/test/morphingdb_test/models/deberta_reader.pt', map_location='cpu')
deberta_reader.eval()
print(f"Loaded deberta_reader on CPU")

# Print start time
start_ms = time.perf_counter() * 1000.0
print(f"EXEC_START_MS={start_ms:.2f}")

# Stage 1: score each of the 5 input vectors with cross_encoder
# Models expect (N, 2, L) tensor: [input_ids, attention_mask]
print("\n=== Stage 1: Cross-encoder scoring ===")
cross_scores = []
for i, vec in enumerate(test_vecs):
    input_ids = vec[0:1, :]       # shape (1, 128)
    attention_mask = vec[1:2, :]  # shape (1, 128)
    inp = np.concatenate([input_ids, attention_mask], axis=0)  # (2, 128)
    inp_tensor = torch.from_numpy(inp).unsqueeze(0).float()     # (1, 2, 128)
    with torch.no_grad():
        logits = cross_encoder(inp_tensor)
    score = logits.squeeze().item()
    cross_scores.append(score)
    print(f"  Sample {i}: cross_encoder score = {score:.4f}")

# Stage 2: for each sample, run deberta_reader on its 2 step2 vectors
print("\n=== Stage 2: DeBERTa reader inference ===")
reader_outputs = []
for i in range(5):
    # Each sample uses 2 step2 vectors: indices i*2 and i*2+1
    s2_a = step2_vecs[i * 2]
    s2_b = step2_vecs[i * 2 + 1]
    
    # Prepare inputs: (2, 2, 128) for 2 passages
    inp_list = []
    for vec in [s2_a, s2_b]:
        input_ids = vec[0:1, :]
        attention_mask = vec[1:2, :]
        combined = np.concatenate([input_ids, attention_mask], axis=0)  # (2, 128)
        inp_list.append(combined)
    inp_arr = np.stack(inp_list, axis=0)  # (2, 2, 128)
    inp_tensor = torch.from_numpy(inp_arr).float()  # (2, 2, 128)
    
    with torch.no_grad():
        reader_out = deberta_reader(inp_tensor)
    # reader_out shape: (2, 2, hidden) -> [start_logits, end_logits]
    
    # Extract token IDs from the output: argmax on start and end logits
    start_logits = reader_out[:, 0, :]  # (2, hidden)
    end_logits = reader_out[:, 1, :]    # (2, hidden)
    
    start_ids = torch.argmax(start_logits, dim=1).tolist()
    end_ids = torch.argmax(end_logits, dim=1).tolist()
    
    # Create token ID sequences for each passage
    ids_for_sample = []
    for j in range(2):
        # Generate a range of token IDs from start to end
        s, e = start_ids[j], end_ids[j]
        if s <= e:
            token_range = list(range(int(s), int(e) + 1))
        else:
            token_range = [int(s)]
        ids_for_sample.append(token_range)
    
    reader_outputs.append(ids_for_sample)
    print(f"  Sample {i}: passage 0 ids=[{start_ids[0]}..{end_ids[0]}], passage 1 ids=[{start_ids[1]}..{end_ids[1]}]")

# Stage 3: decode token IDs using reader_decoder.py
print("\n=== Stage 3: Decode answers ===")
decoded_answers = []
for i, ids_for_sample in enumerate(reader_outputs):
    # Format: "id1,id2,id3;id4,id5,id6" (semicolon separates passages)
    ids_string = ";".join(",".join(str(x) for x in ids) for ids in ids_for_sample)
    print(f"  Sample {i}: calling decoder with --ids '{ids_string[:80]}...'")
    
    result = subprocess.run(
        ['uv', 'run', '/home/why/dbagent/pgdl/test/tools/reader_decoder.py', '--ids', ids_string],
        cwd='/home/why/dbagent/pgdl/test/tools',
        capture_output=True, text=True, timeout=60
    )
    
    if result.returncode == 0:
        decoded_answers.append(result.stdout.strip())
        print(f"  Sample {i}: decoded successfully")
    else:
        decoded_answers.append(f"ERROR: {result.stderr}")
        print(f"  Sample {i}: FAILED - {result.stderr[:200]}")

# Print end time
end_ms = time.perf_counter() * 1000.0
print(f"\nEXEC_END_MS={end_ms:.2f}")

# Print summary
successful = sum(1 for a in decoded_answers if not a.startswith("ERROR"))
print(f"\nDecoded {successful} out of {len(decoded_answers)} answers")
print(f"Total time: {end_ms - start_ms:.2f} ms")
