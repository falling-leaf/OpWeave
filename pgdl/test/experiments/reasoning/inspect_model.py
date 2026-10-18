import torch
import numpy as np

model = torch.jit.load("/home/why/dbagent/pgdl/test/morphingdb_test/models/cross_encoder.pt", map_location="cpu")
print(type(model))
print(model.graph)
print("---")

# Try (1, 2, 128) batch of 1, 2 rows, 128 dim
x = torch.randn(1, 2, 128)
try:
    out = model(x)
    print(f"Shape (1,2,128) worked, output shape: {out.shape if hasattr(out, 'shape') else type(out)}")
except Exception as e:
    print(f"Shape (1,2,128) failed: {e}")

# Try (2, 128)
x2 = torch.randn(2, 128)
try:
    out = model(x2)
    print(f"Shape (2,128) worked, output shape: {out.shape if hasattr(out, 'shape') else type(out)}")
except Exception as e:
    print(f"Shape (2,128) failed: {e}")

# Try (1, 128)
x3 = torch.randn(1, 128)
try:
    out = model(x3)
    print(f"Shape (1,128) worked, output shape: {out.shape if hasattr(out, 'shape') else type(out)}")
except Exception as e:
    print(f"Shape (1,128) failed: {e}")

# Try (1, 4, 128)
x4 = torch.randn(1, 4, 128)
try:
    out = model(x4)
    print(f"Shape (1,4,128) worked, output shape: {out.shape if hasattr(out, 'shape') else type(out)}")
except Exception as e:
    print(f"Shape (1,4,128) failed: {e}")
