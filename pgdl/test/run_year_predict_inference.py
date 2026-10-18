import torch
import pandas as pd

# Load TorchScript model on CPU
model = torch.jit.load("/home/why/baseline/models/year_predict.pt", map_location="cpu")
model.eval()

# Read first 1000 rows of CSV
df = pd.read_csv(
    "/home/why/baseline/data/series/yead_predict/YearPredictionMSD.csv",
    nrows=1000,
)

# Drop 'value90' column
df = df.drop(columns=["value90"])

# Use remaining 90 columns as float32 features
features = df.values.astype("float32")
print(f"Features shape: {features.shape}")

# Batched inference
batch_size = 32
all_outputs = []
for i in range(0, len(features), batch_size):
    batch = torch.tensor(features[i : i + batch_size], dtype=torch.float32)
    with torch.no_grad():
        output = model(batch)
    all_outputs.append(output)

# Concatenate and print output shape
combined = torch.cat(all_outputs, dim=0)
print(f"Output shape: {combined.shape}")
print(f"First 5 outputs: {combined[:5]}")
