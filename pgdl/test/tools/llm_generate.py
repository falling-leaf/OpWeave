import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

# =========================
# Model path
# =========================
MODEL_PATH = "/opt/opweave/pgdl/test/morphingdb_test/models/flan-t5-large"

# =========================
# Command-line argument parsing
# =========================
parser = argparse.ArgumentParser(description='Generate text using FLAN-T5 model')
parser.add_argument('--input', type=str, required=True, help='Input text for the model')
args = parser.parse_args()

# =========================
# Load tokenizer & model
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# Force CPU usage
device = torch.device("cpu")
model = model.to(device)
model.eval()

# =========================
# Build input
# =========================
input_text = args.input

# tokenize
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# =========================
# Inference (generate answer)
# =========================
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=32,
        num_beams=4,          # beam search improves quality
        early_stopping=True
    )

# decode
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# Output only the model output
# =========================
print(answer)