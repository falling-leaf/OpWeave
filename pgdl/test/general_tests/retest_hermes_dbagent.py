#!/usr/bin/env python3
"""
Targeted re-test: Hermes and DBAgent on all 4 datasets.
Updates hermes/dbagent entries in main_result_benchmark_all.json in-place.
Methodology: 1 warm-up + 3 timed runs, report min.
"""

import os, sys, time, json, gc
import numpy as np
import torch
import psycopg2
import pandas as pd
import cv2
from glob import glob
from datetime import datetime

BASE_DIR   = "/home/why/baseline"
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
RESULTS_FILE = "/home/why/dbagent/pgdl/paper/figures/main_result_benchmark_all.json"

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "why", "password": "123456", "dbname": "postgres",
}

N_VALUES   = [2000, 4000, 6000, 8000, 10000]
TIMEOUT_S  = 3600
BATCH_SIZE = 32
GPU_SYNC   = torch.cuda.is_available()
DATASETS   = ["slice", "sst-2", "imagenet", "cifar-10"]

# ── DBAgent ──────────────────────────────────────────────────────────────────
DBAgent_QUERIES = {
    "slice":    "SELECT unnest(db_agent_single('series', sub_table.data)) AS score FROM (SELECT * FROM slice_test LIMIT {n}) AS sub_table;",
    "sst-2":    "SELECT unnest(db_agent_single('nlp', sub_table.comment_vec)) AS score FROM (SELECT * FROM nlp_vector_test LIMIT {n}) AS sub_table;",
    "imagenet": "SELECT unnest(db_agent_single('image_classification', sub_table.image_vector)) AS score FROM (SELECT * FROM imagenet_image_vector_table LIMIT {n}) AS sub_table;",
    "cifar-10": "SELECT unnest(db_agent_single('image_classification', sub_table.image_vector)) AS score FROM (SELECT * FROM cifar_image_vector_table LIMIT {n}) AS sub_table;",
}

def timed_sql(query, timeout=TIMEOUT_S):
    try:
        conn = psycopg2.connect(**DB_CONFIG, connect_timeout=10)
        conn.set_session(readonly=False, autocommit=True)
        cur = conn.cursor()
        cur.execute(f"SET statement_timeout = '{timeout * 1000}';")
        cur.execute("SELECT register_process();")
        gc.collect()
        if GPU_SYNC:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        cur.execute(query)
        rows = cur.fetchall()
        if GPU_SYNC:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        conn.close()
        return elapsed, len(rows)
    except Exception as e:
        print(f"\n    SQL ERROR: {type(e).__name__}: {str(e)[:200]}")
        return None, 0

def benchmark_dbagent(dataset, n):
    query = DBAgent_QUERIES[dataset].format(n=n)
    print(f"    DBAgent warm-up (n={n})...", end=" ", flush=True)
    warm, r = timed_sql(query)
    if warm is None:
        return None
    print(f"{warm:.3f}s (rows={r})")
    times = []
    for i in range(3):
        print(f"    DBAgent run {i+1}/3 (n={n})...", end=" ", flush=True)
        t, r = timed_sql(query)
        if t is None:
            return None
        times.append(t)
        print(f"{t:.3f}s (rows={r})")
    result = min(times)
    print(f"    => DBAgent n={n}: {result:.3f}s")
    return result

# ── Hermes (pure Python inference, same as benchmark_all_baselines.py) ───────
MODEL_PATHS = {
    "slice":    "slice.pt",
    "sst-2":    "traced_albert_vec.pt",
    "imagenet": "resnet18_imagenet.pt",
    "cifar-10": "googlenet_cifar10.pt",
}

IMAGE_SIZE    = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_slice_data(n_rows):
    path = os.path.join(DATA_DIR, "series", "slice", "slice_localization_data.csv")
    df = pd.read_csv(path, nrows=n_rows)
    return df.iloc[:, 1:-1].values.astype(np.float32)

class SimpleWordTokenizer:
    def __init__(self, max_length=128):
        self.max_length = max_length
        self._vocab = {}
        self._counter = 4

    def _get_id(self, word):
        if word not in self._vocab:
            self._vocab[word] = 4 + (self._counter % 29996)
            self._counter += 1
        return self._vocab[word]

    def tokenize_batch(self, texts):
        ml = self.max_length
        ids_list, mask_list = [], []
        for text in texts:
            words = text.lower().split()
            ids  = [2] + [self._get_id(w) for w in words[:ml - 2]] + [3]
            mask = [1] * len(ids)
            if len(ids) < ml:
                ids  += [0] * (ml - len(ids))
                mask += [0] * (ml - len(mask))
            else:
                ids = ids[:ml]; mask = mask[:ml]
            ids_list.append(ids); mask_list.append(mask)
        return {
            "input_ids":      torch.tensor(ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(mask_list, dtype=torch.long),
            "token_type_ids": torch.zeros(len(texts), ml, dtype=torch.long),
            "position_ids":   torch.arange(0, ml).unsqueeze(0).expand(len(texts), -1),
        }

def load_sst2_texts(n_rows):
    path = os.path.join(DATA_DIR, "text", "sst2", "data", "train.tsv")
    df = pd.read_csv(path, sep="\t")
    return df["sentence"].tolist()[:n_rows]

def preprocess_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = cv2.resize(image, IMAGE_SIZE)
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    for i in range(3):
        tensor[i] = (tensor[i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]
    return tensor.unsqueeze(0)

def collect_imagenet_paths(n_rows):
    d = os.path.join(DATA_DIR, "image", "image-net", "data")
    return sorted(glob(os.path.join(d, "*.JPEG")))[:n_rows]

def collect_cifar10_paths(n_rows):
    paths = []
    classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    per = (n_rows // 10) + 1
    for cls in classes:
        for split in ["train", "test"]:
            d = os.path.join(DATA_DIR, "image", "cifar10", split, cls)
            if not os.path.isdir(d):
                continue
            paths.extend(sorted(glob(os.path.join(d, "*.png")))[:per])
    return paths[:n_rows]

def infer_series(model, data):
    outs = []
    with torch.no_grad():
        for i in range(0, data.shape[0], BATCH_SIZE):
            batch = torch.from_numpy(data[i:i+BATCH_SIZE])
            outs.append(model(batch))
    return torch.cat(outs, dim=0)

def infer_albert(model, texts, tokenizer):
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            b_texts = texts[i:i+BATCH_SIZE]
            enc = tokenizer.tokenize_batch(b_texts)
            if GPU_SYNC:
                enc = {k: v.cuda() for k, v in enc.items()}
            out = model(enc["input_ids"], enc["attention_mask"],
                        enc["token_type_ids"], enc["position_ids"])
            outs.append(out[0].cpu())
    return torch.cat(outs, dim=0)

def infer_images(model, paths):
    outs = []
    with torch.no_grad():
        for i in range(0, len(paths), BATCH_SIZE):
            b_paths = paths[i:i+BATCH_SIZE]
            tensors = [preprocess_image(p) for p in b_paths]
            batch   = torch.cat(tensors, dim=0)
            if GPU_SYNC:
                batch = batch.cuda()
            out = model(batch)
            outs.append(out.cpu())
    return torch.cat(outs, dim=0) if outs else torch.empty(0)

def benchmark_hermes(dataset, n):
    model_path = os.path.join(MODELS_DIR, MODEL_PATHS[dataset])
    device = "cpu" if dataset == "slice" else ("cuda" if GPU_SYNC else "cpu")
    print(f"    Hermes: loading {MODEL_PATHS[dataset]} on {device}...")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    if dataset == "slice":
        data = load_slice_data(n)
        def run_infer():
            if GPU_SYNC: torch.cuda.synchronize()
            t0 = time.perf_counter()
            infer_series(model, data)
            if GPU_SYNC: torch.cuda.synchronize()
            return time.perf_counter() - t0
    elif dataset == "sst-2":
        tokenizer = SimpleWordTokenizer()
        texts = load_sst2_texts(n)
        def run_infer():
            if GPU_SYNC: torch.cuda.synchronize()
            t0 = time.perf_counter()
            infer_albert(model, texts, tokenizer)
            if GPU_SYNC: torch.cuda.synchronize()
            return time.perf_counter() - t0
    else:
        paths = collect_imagenet_paths(n) if dataset == "imagenet" else collect_cifar10_paths(n)
        def run_infer():
            if GPU_SYNC: torch.cuda.synchronize()
            t0 = time.perf_counter()
            infer_images(model, paths)
            if GPU_SYNC: torch.cuda.synchronize()
            return time.perf_counter() - t0

    print(f"    Hermes warm-up (3 runs)...")
    for _ in range(3):
        run_infer()

    times = []
    for i in range(3):
        print(f"    Hermes run {i+1}/3 (n={n})...", end=" ", flush=True)
        t = run_infer()
        times.append(t)
        print(f"{t:.3f}s")
    result = min(times)
    print(f"    => Hermes n={n}: {result:.3f}s")
    return result

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"  RETEST: Hermes + DBAgent — {datetime.now().isoformat()}")
    print("=" * 70)
    print(f"  Datasets: {DATASETS}")
    print(f"  n values: {N_VALUES}")
    print(f"  GPU sync: {GPU_SYNC}")
    print("=" * 70)

    # Load existing results
    with open(RESULTS_FILE) as f:
        all_results = json.load(f)

    new_results = {}

    for dataset in DATASETS:
        new_results[dataset] = {}
        print(f"\n{'#'*60}")
        print(f"# DATASET: {dataset}")
        print(f"{'#'*60}")

        for n in N_VALUES:
            n_key = str(n)
            new_results[dataset][n] = {}
            print(f"\n  --- n={n} ---")

            # DBAgent
            print(f"  [DBAgent]")
            try:
                t = benchmark_dbagent(dataset, n)
                new_results[dataset][n]["dbagent"] = t
            except Exception as e:
                import traceback; traceback.print_exc()
                new_results[dataset][n]["dbagent"] = None

            # Hermes
            print(f"  [Hermes]")
            try:
                t = benchmark_hermes(dataset, n)
                new_results[dataset][n]["hermes"] = t
            except Exception as e:
                import traceback; traceback.print_exc()
                new_results[dataset][n]["hermes"] = None

            # Report
            da = new_results[dataset][n]["dbagent"]
            hm = new_results[dataset][n]["hermes"]
            da_s = f"{da:.3f}s" if da else "FAIL"
            hm_s = f"{hm:.3f}s" if hm else "FAIL"
            print(f"  => n={n}: dbagent={da_s}, hermes={hm_s}")

            # Patch into existing results
            if n_key not in all_results[dataset]:
                all_results[dataset][n_key] = {}
            if da is not None:
                all_results[dataset][n_key]["dbagent"] = da
            if hm is not None:
                all_results[dataset][n_key]["hermes"] = hm

    # Save updated results
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Summary table
    print(f"\n{'='*70}")
    print("  SUMMARY (seconds, min of 3 runs)")
    print(f"{'='*70}")
    for dataset in DATASETS:
        print(f"\n{dataset}:")
        print(f"  {'n':<8}  {'dbagent':<14} {'hermes':<14}")
        print("  " + "-" * 36)
        for n in N_VALUES:
            da = new_results[dataset][n].get("dbagent")
            hm = new_results[dataset][n].get("hermes")
            da_s = f"{da:.4f}" if da is not None else "FAIL"
            hm_s = f"{hm:.4f}" if hm is not None else "FAIL"
            print(f"  {n:<8}  {da_s:<14} {hm_s:<14}")

    # Also save new results separately for easy diffing
    new_file = RESULTS_FILE.replace(".json", "_hermes_dbagent_retest.json")
    with open(new_file, "w") as f:
        json.dump(new_results, f, indent=2, default=str)
    print(f"\nNew-only results also saved to {new_file}")

if __name__ == "__main__":
    main()
