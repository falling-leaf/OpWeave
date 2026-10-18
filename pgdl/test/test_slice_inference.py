import torch
import pandas as pd

# Load model on CPU
model = torch.jit.load('/home/why/baseline/models/slice.pt')
model.eval()

# Read exactly 2000 rows with header=0
df = pd.read_csv('/home/why/baseline/data/series/slice/slice_localization_data.csv', header=0, nrows=2000)
print(f"DataFrame shape: {df.shape}")

# Use columns at positions 1 through -2 inclusive (exclude first and last)
features = df.iloc[:, 1:-1].values.astype('float32')
print(f"Features shape: {features.shape}")

# Convert to tensor
X = torch.from_numpy(features)

# Batched inference with batch_size=32
batch_size = 32
outputs = []
with torch.no_grad():
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        out = model(batch)
        outputs.append(out)

result = torch.cat(outputs, dim=0)
print(f"Output shape: {result.shape}")
