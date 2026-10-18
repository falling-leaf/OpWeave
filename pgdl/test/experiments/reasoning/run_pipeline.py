import time
import numpy as np
import torch
import sys
import subprocess
import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(host='localhost', port=5432, user='why', password='123456', dbname='postgres')
cur = conn.cursor()

# Load 20 rows from musique_vector_test
cur.execute('SELECT get_mvec_data(reasoning_vec) FROM musique_vector_test ORDER BY id LIMIT 20')
test_rows = cur.fetchall()
test_data = []
for row in test_rows:
    arr = np.array(row[0], dtype=np.float64).reshape(4, 128)
    test_data.append(arr)

# Load 40 rows from musique_vector_step2
cur.execute('SELECT get_mvec_data(text_vec) FROM musique_vector_step2 ORDER BY id LIMIT 40')
step2_rows = cur.fetchall()
step2_data = []
for row in step2_rows:
    arr = np.array(row[0], dtype=np.float64).reshape(4, 128)
    step2_data.append(arr)

cur.close()
conn.close()

# Load models on CPU
cross_encoder = torch.jit.load('/home/why/dbagent/pgdl/test/morphingdb_test/models/cross_encoder.pt', map_location='cpu')
deberta_reader = torch.jit.load('/home/why/dbagent/pgdl/test/morphingdb_test/models/deberta_reader.pt', map_location='cpu')

# Print EXEC_START_MS
exec_start_ms = time.perf_counter() * 1000.0
print(f'EXEC_START_MS={exec_start_ms}')

# Stage 1: Score each of the 20 input vectors (using first 2 rows) with cross_encoder
print('\n=== Stage 1: Cross-Encoder Scoring ===')
cross_scores = []
for i in range(20):
    arr = test_data[i]
    # Use first 2 rows: row 0 = input_ids, row 1 = mask
    tensor_input = torch.tensor(arr[:2], dtype=torch.long).unsqueeze(0)  # (1, 2, 128)
    score = cross_encoder(tensor_input)
    cross_scores.append(score.item())
    if i < 5 or i >= 18:
        print(f'  Sample {i}: score = {score.item():.4f}')
    elif i == 5:
        print(f'  ...')

print(f'  Total samples scored: {len(cross_scores)}')

# Stage 2: For each sample, run deberta_reader on its 2 step2 vectors
print('\n=== Stage 2: DeBERTa Reader ===')
all_decoded_results = []
decoded_count = 0

for i in range(20):
    # Each sample gets 2 step2 vectors (40 total / 20 samples = 2 per sample)
    s2_idx_1 = i * 2
    s2_idx_2 = i * 2 + 1
    
    if s2_idx_2 >= len(step2_data):
        break
    
    s2_arr1 = step2_data[s2_idx_1]
    s2_arr2 = step2_data[s2_idx_2]
    
    # Stack both step2 vectors: (2, 2, 128) - batch of 2
    batch_input = torch.tensor(
        np.stack([s2_arr1[:2], s2_arr2[:2]]),
        dtype=torch.long
    )  # (2, 2, 128)
    
    # Run deberta_reader -> returns (batch, 2, 128) = [start_logits, end_logits]
    reader_out = deberta_reader(batch_input)
    
    # Extract token IDs from start/end positions
    for j in range(batch_input.shape[0]):
        input_ids = batch_input[j, 0, :].tolist()  # token IDs
        start_logits = reader_out[j, 0, :]
        end_logits = reader_out[j, 1, :]
        
        start_pos = torch.argmax(start_logits).item()
        end_pos = torch.argmax(end_logits).item()
        
        # Ensure valid range
        if end_pos < start_pos:
            start_pos, end_pos = end_pos, start_pos
        
        # Extract token IDs for the answer span
        answer_ids = input_ids[start_pos:end_pos + 1]
        
        if answer_ids:
            ids_string = ','.join(str(int(x)) for x in answer_ids)
            all_decoded_results.append({
                'sample': i,
                'vec': j + 1,
                'start': start_pos,
                'end': end_pos,
                'ids_string': ids_string
            })

# Stage 3: Decode token IDs via subprocess
print(f'\n=== Stage 3: Decoding ({len(all_decoded_results)} items to decode) ===')
decoded_answers = 0

for item in all_decoded_results:
    try:
        result = subprocess.run(
            ['uv', 'run', '/home/why/dbagent/pgdl/test/tools/reader_decoder.py', '--ids', item['ids_string']],
            cwd='/home/why/dbagent/pgdl/test/tools',
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            decoded_answers += 1
            # Print first few decoded results
            if decoded_answers <= 5:
                print(f'  Sample {item["sample"]}, vec {item["vec"]}: decoded OK')
                # Show the answer text
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if '"paragraph_' in line:
                        print(f'    {line.strip()}')
    except subprocess.TimeoutExpired:
        print(f'  Sample {item["sample"]}, vec {item["vec"]}: TIMEOUT')
    except Exception as e:
        print(f'  Sample {item["sample"]}, vec {item["vec"]}: ERROR - {e}')

# Print EXEC_END_MS
exec_end_ms = time.perf_counter() * 1000.0
print(f'\nEXEC_END_MS={exec_end_ms}')
print(f'\nTotal answers decoded: {decoded_answers}')
print(f'Total elapsed: {(exec_end_ms - exec_start_ms):.2f} ms')
