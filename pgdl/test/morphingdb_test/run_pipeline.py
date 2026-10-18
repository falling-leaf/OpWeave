import time
import numpy as np
import torch
import sys
import subprocess
import psycopg2
import json

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    user="why",
    password="123456",
    dbname="postgres"
)
cur = conn.cursor()

# Load 5 rows from musique_vector_test
cur.execute("""
    SELECT get_mvec_data(reasoning_vec)
    FROM musique_vector_test
    ORDER BY id LIMIT 5
""")
rows_test = cur.fetchall()
print(f"Loaded {len(rows_test)} rows from musique_vector_test")

# Load 10 rows from musique_vector_step2
cur.execute("""
    SELECT get_mvec_data(text_vec)
    FROM musique_vector_step2
    ORDER BY id LIMIT 10
""")
rows_step2 = cur.fetchall()
print(f"Loaded {len(rows_step2)} rows from musique_vector_step2")

cur.close()
conn.close()

# Convert to numpy arrays and reshape to (4, 128)
vecs_test = [np.array(r[0]).reshape(4, 128) for r in rows_test]
vecs_step2 = [np.array(r[0]).reshape(4, 128) for r in rows_step2]

print(f"Test vec shape: {vecs_test[0].shape}")
print(f"Step2 vec shape: {vecs_step2[0].shape}")

# Load models
cross_encoder = torch.jit.load(
    '/home/why/dbagent/pgdl/test/morphingdb_test/models/cross_encoder.pt',
    map_location='cpu'
)
cross_encoder.eval()
print("cross_encoder loaded")

deberta_reader = torch.jit.load(
    '/home/why/dbagent/pgdl/test/morphingdb_test/models/deberta_reader.pt',
    map_location='cpu'
)
deberta_reader.eval()
print("deberta_reader loaded")

# Print start time
print(f"EXEC_START_MS={time.perf_counter() * 1000.0}")

# Stage 1: Score each of the 5 input vectors (using first 2 rows) with cross_encoder
print("\n--- Stage 1: Cross-Encoder Scoring ---")
stage1_scores = []
for i in range(min(2, len(vecs_test))):  # Use first 2 rows as specified
    vec = vecs_test[i]
    # Extract input_ids (row 0) and mask (row 1), add batch dim
    input_ids = torch.tensor(vec[0:1], dtype=torch.long)  # (1, 128)
    mask = torch.tensor(vec[1:2], dtype=torch.long)        # (1, 128)
    # Stack into (1, 2, 128) for model input
    model_input = torch.cat([input_ids, mask], dim=0).unsqueeze(0)  # (1, 2, 128)
    with torch.no_grad():
        score = cross_encoder(model_input)
    stage1_scores.append(score)
    print(f"  Sample {i}: score shape={score.shape}, value={score}")

# Stage 2: For each sample, run deberta_reader on its 2 step2 vectors
print("\n--- Stage 2: DeBERTa Reader Inference ---")
stage2_outputs = []
for i in range(min(2, len(vecs_test))):
    # Use 2 step2 vectors per sample (step2 rows i*2 and i*2+1)
    sample_outputs = []
    for j in range(2):
        idx = i * 2 + j
        if idx < len(vecs_step2):
            vec = vecs_step2[idx]
            input_ids = torch.tensor(vec[0:1], dtype=torch.long)
            mask = torch.tensor(vec[1:2], dtype=torch.long)
            model_input = torch.cat([input_ids, mask], dim=0).unsqueeze(0)
            with torch.no_grad():
                output = deberta_reader(model_input)
            sample_outputs.append(output)
            print(f"  Sample {i}, step2[{j}]: output shape={output.shape}")
    stage2_outputs.append(sample_outputs)

# Stage 3: Decode token IDs using reader_decoder.py
print("\n--- Stage 3: Decoding ---")
decoded_answers = []
for i, sample_outputs in enumerate(stage2_outputs):
    for j, output in enumerate(sample_outputs):
        # deberta_reader outputs (1, 2, 128) = start_logits, end_logits
        # Extract start/end positions via argmax
        logits = output.squeeze(0)  # (2, 128)
        start_pos = torch.argmax(logits[0, :]).item()
        end_pos = torch.argmax(logits[1, :]).item()
        if end_pos < start_pos:
            end_pos = start_pos

        # Get the input_ids from the corresponding step2 vector
        idx = i * 2 + j
        if idx < len(vecs_step2):
            input_ids_row = vecs_step2[idx][0].astype(int)  # row 0 = input_ids
            # Extract answer span token IDs
            answer_ids = input_ids_row[start_pos:end_pos + 1].tolist()
            # Filter out zero/padding tokens
            answer_ids = [int(x) for x in answer_ids if int(x) > 0]
            ids_string = ",".join(str(x) for x in answer_ids)

            # Call subprocess for decoding
            if ids_string:
                result = subprocess.run(
                    ["uv", "run", "/home/why/dbagent/pgdl/test/tools/reader_decoder.py", "--ids", ids_string],
                    cwd="/home/why/dbagent/pgdl/test/tools",
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                try:
                    # The decoder outputs multi-line indented JSON + timing line
                    # Find the JSON block (from first { to last })
                    output_text = result.stdout.strip()
                    # Remove the "Execution time:" line at the end
                    lines = output_text.split('\n')
                    json_lines = []
                    in_json = False
                    for line in lines:
                        if line.strip().startswith('{'):
                            in_json = True
                        if in_json:
                            json_lines.append(line)
                        if line.strip().endswith('}'):
                            break
                    decoded = json.loads('\n'.join(json_lines))
                    decoded_answers.append(decoded)
                    # Show first decoded paragraph
                    first_val = list(decoded.values())[0] if decoded else "(empty)"
                    print(f"  Sample {i}, step2[{j}]: start={start_pos}, end={end_pos}, answer='{first_val[:80]}'")
                except (json.JSONDecodeError, IndexError) as e:
                    print(f"  Sample {i}, step2[{j}]: decode failed ({e}): stdout={result.stdout[:200]} stderr={result.stderr[:200]}")
            else:
                print(f"  Sample {i}, step2[{j}]: no valid IDs to decode (start={start_pos}, end={end_pos})")

# Print end time
print(f"\nEXEC_END_MS={time.perf_counter() * 1000.0}")
print(f"Decoded {len(decoded_answers)} answers")
