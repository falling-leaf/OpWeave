#!/usr/bin/env python3
"""Hermes cifar10 execution, same path as the main-table measurement.

benchmark_all_baselines.benchmark_hermes reads the PNG files, decodes and
normalizes them, then runs googlenet in batches of 32. That whole step is the
timed latency (7.148s at n=2000). The earlier SQL path, get_mvec_data of the
224x224 vectors, spent about 115s fetching 2000 rows, which is not what the
main table measures.

load_s stays 0 so compare_baselines.py, which reports avg + load_s, keeps the
main-table number. Model loading is outside the timer.
"""
import argparse
import json
import os
import time
from glob import glob

import cv2
import torch

MODEL_PATH = "/home/why/baseline/models/googlenet_cifar10.pt"
DATA_DIR = "/home/why/baseline/data/image/cifar10"
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def collect_paths(n):
    per = (n // 10) + 1
    paths = []
    for cls in CLASSES:
        for split in ("train", "test"):
            folder = os.path.join(DATA_DIR, split, cls)
            if os.path.isdir(folder):
                paths.extend(sorted(glob(os.path.join(folder, "*.png")))[:per])
    return paths[:n]


def preprocess(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"failed to read {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    image = cv2.resize(image, IMAGE_SIZE)
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    for channel in range(3):
        tensor[channel] = (tensor[channel] - MEAN[channel]) / STD[channel]
    return tensor.unsqueeze(0)


def run_inference(paths, model, device):
    with torch.no_grad():
        for start in range(0, len(paths), BATCH_SIZE):
            batch = torch.cat([preprocess(p) for p in paths[start:start + BATCH_SIZE]], 0)
            model(batch.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-count", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--json-output", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()
    paths = collect_paths(args.row_count)
    if len(paths) < args.row_count:
        raise SystemExit(f"only {len(paths)} images for n={args.row_count}")

    run_inference(paths[:BATCH_SIZE], model, device)

    times = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        run_inference(paths, model, device)
        times.append(time.perf_counter() - start)

    result = {
        "row_count": len(paths),
        "repetitions": args.repetitions,
        "device": str(device),
        "load_s": 0.0,
        "times": times,
        "avg": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
    }
    if args.json_output:
        print(json.dumps(result))
    else:
        print(f"n={len(paths)} device={device} min={result['min']*1000:.1f}ms")


if __name__ == "__main__":
    main()
