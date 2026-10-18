#!/usr/bin/env python3
"""
Comprehensive Benchmark: All 6 Baselines on 4 Datasets
========================================================
Datasets: slice, sst-2, imagenet, cifar-10
n values: 2000, 4000, 6000, 8000, 10000
Baselines: evadb, morphingdb, gendb, hermes, neurdb, DBAgent

Methodology (per baseline):
  - Warm-up run (not counted)
  - 3 timed runs, report min
  - Large timeout: 3600s (1 hour) per query

Hermes/GenDB: Uses existing generated/execution code, just modifies n value.
"""

import os, sys, time, json, gc, subprocess, argparse, csv
import numpy as np
import torch
import psycopg2
import pandas as pd
import cv2
from glob import glob
from datetime import datetime
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = "/home/why/baseline"
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
GENDB_DIR = os.path.join(BASE_DIR, "GenDB/benchmarks")
HERMES_DIR = os.path.join(BASE_DIR, "hermes")
NEURDB_DIR = os.path.join(BASE_DIR, "neurdb")

RESULTS_FILE = "/home/why/dbagent/pgdl/paper/figures/main_result_benchmark_all.json"

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "user": "why", "password": "123456", "dbname": "postgres",
}

N_VALUES = [2000, 4000, 6000, 8000, 10000]
TIMEOUT_S = 3600  # 1 hour per query
BATCH_SIZE = 32
GPU_SYNC = torch.cuda.is_available()

# ─── Helper: safe SQL timing ───────────────────────────────────────────
def timed_sql(query, timeout=TIMEOUT_S):
    """Run a single SQL query and return wall-clock seconds. Returns None on failure."""
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
        return elapsed
    except Exception as e:
        print(f"    SQL ERROR: {type(e).__name__}: {str(e)[:200]}")
        return None

# ═══════════════════════════════════════════════════════════════════════════
# 1. DBAgent (db_agent_single)
# ═══════════════════════════════════════════════════════════════════════════
DBAgent_QUERIES = {
    "slice":   "SELECT unnest(db_agent_single('series', sub_table.data)) AS score FROM (SELECT * FROM slice_test LIMIT {n}) AS sub_table;",
    "sst-2":   "SELECT unnest(db_agent_single('nlp', sub_table.comment_vec)) AS score FROM (SELECT * FROM nlp_vector_test LIMIT {n}) AS sub_table;",
    "imagenet":"SELECT unnest(db_agent_single('image_classification', sub_table.image_vector)) AS score FROM (SELECT * FROM imagenet_image_vector_table LIMIT {n}) AS sub_table;",
    "cifar-10":"SELECT unnest(db_agent_single('image_classification', sub_table.image_vector)) AS score FROM (SELECT * FROM cifar_image_vector_table LIMIT {n}) AS sub_table;",
}

def benchmark_dbagent(dataset, n):
    query = DBAgent_QUERIES[dataset].format(n=n)
    print(f"    DBAgent warm-up (n={n})...")
    warm = timed_sql(query)
    if warm is None:
        return None
    print(f"      warm-up: {warm:.3f}s")
    times = []
    for i in range(3):
        print(f"    DBAgent run {i+1}/3 (n={n})...", end=" ", flush=True)
        t = timed_sql(query)
        if t is None:
            return None
        times.append(t)
        print(f"{t:.3f}s")
    return min(times)

# ═══════════════════════════════════════════════════════════════════════════
# 2. MorphingDB (predict_batch_float8)
# ═══════════════════════════════════════════════════════════════════════════
MorphingDB_QUERIES = {
    "slice":   "SELECT predict_batch_float8('slice', 'cpu', data) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM slice_test LIMIT {n};",
    "sst-2":   "SELECT predict_batch_float8('sst2_vec', 'gpu', comment_vec) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM nlp_vector_test LIMIT {n};",
    "imagenet":"SELECT predict_batch_float8('defect_vec', 'gpu', image_vector) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM imagenet_image_vector_table LIMIT {n};",
    "cifar-10":"SELECT predict_batch_float8('googlenet_cifar10', 'gpu', image_vector) OVER (ROWS BETWEEN CURRENT ROW AND 31 FOLLOWING) FROM cifar_image_vector_table LIMIT {n};",
}

def benchmark_morphingdb(dataset, n):
    query = MorphingDB_QUERIES[dataset].format(n=n)
    print(f"    MorphingDB cold run (n={n})...", end=" ", flush=True)
    cold = timed_sql(query)
    if cold is None:
        return None
    print(f"{cold:.3f}s")
    times = []
    for i in range(3):
        print(f"    MorphingDB run {i+1}/3 (n={n})...", end=" ", flush=True)
        t = timed_sql(query)
        if t is None:
            return None
        times.append(t)
        print(f"{t:.3f}s")
    return min(times)

# ═══════════════════════════════════════════════════════════════════════════
# 3. EvaDB
# ═══════════════════════════════════════════════════════════════════════════
import warnings
warnings.filterwarnings("ignore")
import evadb

EvaDB_CONFIG = {
    "database": "postgres", "host": "localhost", "port": "5432",
    "user": "why", "password": "123456"
}

# Need to add test dir to PYTHONPATH for evadb function imports
TEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EvaDB_DATASETS = {
    "slice": {
        "func_name": "SliceFunction",
        "impl_file": os.path.join(TEST_DIR,
                                   "morphingdb_test/series_test/slice_test/evadb_slice.py"),
        "query_sql": "SELECT {func}({cols}) FROM postgres_data.slice_origin_test LIMIT {n};",
        "n_cols": 384,
        "single_col": None,
    },
    "sst-2": {
        "func_name": "SST2Function",
        "impl_file": os.path.join(TEST_DIR,
                                   "morphingdb_test/text_test/sst2/evadb_sst2.py"),
        "query_sql": "SELECT {func}(comment) FROM postgres_data.nlp_test LIMIT {n};",
        "n_cols": None,
        "single_col": "comment",
    },
    "imagenet": {
        "func_name": "ImagenetFunction",
        "impl_file": os.path.join(TEST_DIR,
                                   "morphingdb_test/image_test/imagenet/evadb_resnet18.py"),
        "query_sql": "SELECT {func}(image_path) FROM postgres_data.imagenet_image_table LIMIT {n};",
        "n_cols": None,
        "single_col": "image_path",
    },
    "cifar-10": {
        "func_name": "CifarFunction",
        "impl_file": os.path.join(TEST_DIR,
                                   "morphingdb_test/image_test/cifar10/evadb_cifar.py"),
        "query_sql": "SELECT {func}(image_path) FROM postgres_data.cifar_image_table LIMIT {n};",
        "n_cols": None,
        "single_col": "image_path",
    },
}

def setup_evadb():
    conn = evadb.connect()
    cursor = conn.cursor()
    try:
        cursor.query("DROP DATABASE IF EXISTS postgres_data;").df()
    except:
        pass
    cursor.query(f"""
        CREATE DATABASE IF NOT EXISTS postgres_data
        WITH ENGINE = 'postgres',
        PARAMETERS = {json.dumps(EvaDB_CONFIG)}
    """).df()
    return conn, cursor

def build_cols(n_cols):
    return ", ".join([f"value{i} NDARRAY FLOAT32" for i in range(n_cols)])

def benchmark_evadb(dataset, n):
    ds = EvaDB_DATASETS[dataset]
    # Ensure morphingdb_test is importable (must be in sys.path before evadb imports)
    if TEST_DIR not in sys.path:
        sys.path.insert(0, TEST_DIR)
    print(f"    EvaDB setting up connection...")
    conn, cursor = setup_evadb()

    # Create function
    func_name = ds["func_name"]
    impl_file = ds["impl_file"]
    n_cols = ds.get("n_cols")
    single_col = ds.get("single_col")

    if n_cols:
        col_spec = build_cols(n_cols)
        cols_list = [f"value{i}" for i in range(n_cols)]
        cols = ", ".join(cols_list)
        out_spec = "OUTPUT (labels NDARRAY FLOAT32(1))"
    elif single_col == "comment":
        col_spec = "comment TEXT(128)"
        cols = "comment"
        out_spec = "OUTPUT (labels NDARRAY FLOAT32(3))"
    else:
        # image_path
        col_spec = "image_path TEXT(256)"
        cols = "image_path"
        out_spec = "OUTPUT (labels NDARRAY FLOAT32(1000))"

    func_sql = f"""
        CREATE FUNCTION IF NOT EXISTS {func_name}
        INPUT  ({col_spec})
        {out_spec}
        TYPE Classification
        IMPL  '{impl_file}';
    """
    try:
        cursor.query(func_sql).df()
        print(f"    EvaDB function {func_name} created OK")
    except Exception as e:
        print(f"    EvaDB function creation failed: {str(e)[:100]}")
        return None

    # Warmup
    print(f"    EvaDB warmup (n=10)...")
    warmup_sql = ds["query_sql"].format(func=func_name, cols=cols, n=10)
    try:
        cursor.query(warmup_sql).df()
        print(f"    EvaDB warmup OK")
    except Exception as e:
        print(f"    EvaDB warmup failed: {str(e)[:100]}")
        return None

    # Timed runs
    times = []
    sql = ds["query_sql"].format(func=func_name, cols=cols, n=n)
    for i in range(3):
        print(f"    EvaDB run {i+1}/3 (n={n})...", end=" ", flush=True)
        try:
            if GPU_SYNC:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = cursor.query(sql).df()
            if GPU_SYNC:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"{elapsed:.3f}s (rows={len(result)})")
        except Exception as e:
            print(f"FAILED: {str(e)[:100]}")
            return None
        time.sleep(1)

    return min(times)

# ═══════════════════════════════════════════════════════════════════════════
# 4. Hermes — Python inference (existing benchmark code pattern)
# ═══════════════════════════════════════════════════════════════════════════
MODEL_PATHS = {
    "slice":   "slice.pt",
    "sst-2":   "traced_albert_vec.pt",
    "imagenet":"resnet18_imagenet.pt",
    "cifar-10":"googlenet_cifar10.pt",
}

SEQ_LENGTH = 128
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Data loaders
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
            ids = [2] + [self._get_id(w) for w in words[:ml - 2]] + [3]
            mask = [1] * len(ids)
            if len(ids) < ml:
                ids += [0] * (ml - len(ids))
                mask += [0] * (ml - len(mask))
            else:
                ids = ids[:ml]; mask = mask[:ml]
            ids_list.append(ids); mask_list.append(mask)
        return {
            "input_ids": torch.tensor(ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(mask_list, dtype=torch.long),
            "token_type_ids": torch.zeros(len(texts), ml, dtype=torch.long),
            "position_ids": torch.arange(0, ml).unsqueeze(0).expand(len(texts), -1),
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
        for split in ["train","test"]:
            d = os.path.join(DATA_DIR, "image", "cifar10", split, cls)
            if not os.path.isdir(d): continue
            paths.extend(sorted(glob(os.path.join(d, "*.png")))[:per])
    return paths[:n_rows]

# Inference functions
def infer_series(model, data):
    n = data.shape[0]
    outs = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch = torch.from_numpy(data[i:i+BATCH_SIZE])
            outs.append(model(batch))
    return torch.cat(outs, dim=0)

def infer_albert(model, texts, tokenizer):
    n = len(texts)
    outs = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            b_texts = texts[i:i+BATCH_SIZE]
            enc = tokenizer.tokenize_batch(b_texts)
            if GPU_SYNC:
                enc = {k: v.cuda() for k, v in enc.items()}
            out = model(enc["input_ids"], enc["attention_mask"],
                       enc["token_type_ids"], enc["position_ids"])
            outs.append(out[0].cpu())
    return torch.cat(outs, dim=0)

def infer_images(model, paths):
    n = len(paths)
    outs = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            b_paths = paths[i:i+BATCH_SIZE]
            tensors = [preprocess_image(p) for p in b_paths]
            batch = torch.cat(tensors, dim=0)
            if GPU_SYNC:
                batch = batch.cuda()
            out = model(batch)
            outs.append(out.cpu())
    return torch.cat(outs, dim=0) if outs else torch.empty(0)

def benchmark_hermes(dataset, n):
    """Hermes-style: Pure Python inference with torch.jit.load + batched inference."""
    model_path = os.path.join(MODELS_DIR, MODEL_PATHS[dataset])
    device = "cpu" if dataset == "slice" else ("cuda" if GPU_SYNC else "cpu")
    print(f"    Hermes: loading model {MODEL_PATHS[dataset]} on {device}...")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    tokenizer = None
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
    else:  # image
        if dataset == "imagenet":
            paths = collect_imagenet_paths(n)
        else:
            paths = collect_cifar10_paths(n)
        def run_infer():
            if GPU_SYNC: torch.cuda.synchronize()
            t0 = time.perf_counter()
            infer_images(model, paths)
            if GPU_SYNC: torch.cuda.synchronize()
            return time.perf_counter() - t0

    # Warm-up (3 batches to warm GPU)
    print(f"    Hermes warm-up...")
    for _ in range(3):
        run_infer()
    # 3 timed runs
    times = []
    for i in range(3):
        print(f"    Hermes run {i+1}/3 (n={n})...", end=" ", flush=True)
        t = run_infer()
        times.append(t)
        print(f"{t:.3f}s")
    return min(times)

# ═══════════════════════════════════════════════════════════════════════════
# 5. GenDB — Uses existing benchmark_single_run.py per dataset (execution only)
# ═══════════════════════════════════════════════════════════════════════════
GENDB_SCRIPTS = {
    "slice": os.path.join(GENDB_DIR, "slice-inference/benchmark_single_run.py"),
    "sst-2": os.path.join(GENDB_DIR, "sst2-inference/benchmark_single_run.py"),
    "imagenet": os.path.join(GENDB_DIR, "imagenet-inference/benchmark_single_run.py"),
    "cifar-10": os.path.join(GENDB_DIR, "cifar10-inference/benchmark_single_run.py"),
}

GENDB_DATA = {
    "slice": {
        "subset_dir": os.path.join(GENDB_DIR, "slice-inference/subsets/slice_data_10000"),
        "model": os.path.join(MODELS_DIR, "slice.pt"),
    },
    "sst-2": {
        "csv": os.path.join(GENDB_DIR, "sst2-inference/subsets/sst2_10000.csv"),
        "model": os.path.join(MODELS_DIR, "traced_albert_vec.pt"),
    },
    "imagenet": {
        "csv": os.path.join(GENDB_DIR, "imagenet-inference/subsets/imagenet_10000.csv"),
        "model": os.path.join(MODELS_DIR, "resnet18_imagenet.pt"),
    },
    "cifar-10": {
        "csv": os.path.join(GENDB_DIR, "cifar10-inference/subsets/cifar10_10000.csv"),
        "model": os.path.join(MODELS_DIR, "googlenet_cifar10.pt"),
    },
}

def benchmark_gendb(dataset, n):
    """GenDB execution phase: run existing benchmark_single_run.py with modified n."""
    script = GENDB_SCRIPTS[dataset]
    ds_data = GENDB_DATA[dataset]
    output_dir = f"/tmp/gendb_bench_{dataset}_{n}"
    is_image = dataset in ("imagenet", "cifar-10")

    if dataset == "slice":
        cmd = ["python3", script, ds_data["subset_dir"], ds_data["model"], output_dir, str(n)]
    else:
        cmd = ["python3", script, ds_data["csv"], ds_data["model"], output_dir, str(n)]

    print(f"    GenDB: {cmd}")

    def parse_gendb_time(r, output_dir):
        """Parse timing from GenDB output. Image datasets save JSON to file; others print to stdout."""
        if is_image:
            # Read result.json from output directory
            result_file = os.path.join(output_dir, "result.json")
            if os.path.exists(result_file):
                with open(result_file) as f:
                    j = json.load(f)
                return j.get("timing_ms", {}).get("total", 0) / 1000.0
            return None
        else:
            # Parse JSON from stdout (last line)
            j = json.loads(r.stdout.strip().split('\n')[-1])
            return j.get("total_ms", 0) / 1000.0

    times = []
    # Warm-up
    print(f"    GenDB warm-up (n={n})...")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": "0" if GPU_SYNC and dataset != "slice" else ""})
        if r.returncode != 0:
            print(f"    GenDB warm-up FAILED: {r.stderr[:200]}")
            return None
        try:
            warm_time = parse_gendb_time(r, output_dir)
            if warm_time is None:
                print(f"      warm-up output not parseable: {r.stdout[:200]}")
            else:
                print(f"      warm-up: {warm_time:.3f}s")
        except Exception as e:
            print(f"      warm-up parse error: {e}")
    except subprocess.TimeoutExpired:
        print(f"    GenDB warm-up TIMEOUT")
        return None

    # 3 timed runs
    for i in range(3):
        print(f"    GenDB run {i+1}/3 (n={n})...", end=" ", flush=True)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": "0" if GPU_SYNC and dataset != "slice" else ""})
            if r.returncode != 0:
                print(f"FAILED")
                return None
            elapsed = parse_gendb_time(r, output_dir)
            if elapsed is None:
                print(f"PARSE_ERROR")
                return None
            times.append(elapsed)
            print(f"{elapsed:.3f}s")
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT")
            return None
        except Exception as e:
            print(f"ERROR: {e}")
            return None

    return min(times)

# ═══════════════════════════════════════════════════════════════════════════
# 6. NeurDB — Python inference (NeurDB-style, similar to benchmark_python)
# ═══════════════════════════════════════════════════════════════════════════
def benchmark_neurdb(dataset, n):
    """
    NeurDB inference benchmark.
    NeurDB uses predict_batch_float8 via its own extension for series,
    and Python inference for NLP/Image.
    For consistency and to match paper methodology, we use pure Python inference
    similar to the neurdb benchmark pattern (torch.jit.load + batch inference).
    """
    # NeurDB: same inference pattern as Hermes but with neurdb-specific settings
    model_path = os.path.join(MODELS_DIR, MODEL_PATHS[dataset])
    device = "cpu" if dataset == "slice" else ("cuda" if GPU_SYNC else "cpu")
    print(f"    NeurDB: loading model {MODEL_PATHS[dataset]} on {device}...")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    tokenizer = None
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
        if dataset == "imagenet":
            paths = collect_imagenet_paths(n)
        else:
            paths = collect_cifar10_paths(n)
        def run_infer():
            if GPU_SYNC: torch.cuda.synchronize()
            t0 = time.perf_counter()
            infer_images(model, paths)
            if GPU_SYNC: torch.cuda.synchronize()
            return time.perf_counter() - t0

    # Warm-up
    print(f"    NeurDB warm-up...")
    for _ in range(3):
        run_infer()
    # 3 timed runs
    times = []
    for i in range(3):
        print(f"    NeurDB run {i+1}/3 (n={n})...", end=" ", flush=True)
        t = run_infer()
        times.append(t)
        print(f"{t:.3f}s")
    return min(times)

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print(f"  ALL BASELINES BENCHMARK — {datetime.now().isoformat()}")
    print("=" * 70)
    print(f"  Datasets: slice, sst-2, imagenet, cifar-10")
    print(f"  n values: {N_VALUES}")
    print(f"  GPU sync: {GPU_SYNC}")
    print(f"  Timeout: {TIMEOUT_S}s per query")
    print(f"  Baselines: evadb, morphingdb, gendb, hermes, neurdb, DBAgent")
    print("=" * 70)

    BASELINES = [
        ("evadb", benchmark_evadb),
        ("morphingdb", benchmark_morphingdb),
        ("gendb", benchmark_gendb),
        ("hermes", benchmark_hermes),
        ("neurdb", benchmark_neurdb),
        ("dbagent", benchmark_dbagent),
    ]

    all_results = {}

    for dataset in ["slice", "sst-2", "imagenet", "cifar-10"]:
        all_results[dataset] = {}
        print(f"\n{'#'*60}")
        print(f"# DATASET: {dataset}")
        print(f"{'#'*60}")

        for n in N_VALUES:
            print(f"\n  --- n={n} ---")
            all_results[dataset][n] = {}

            for bl_name, bl_func in BASELINES:
                print(f"  [{bl_name}]")
                try:
                    t = bl_func(dataset, n)
                    all_results[dataset][n][bl_name] = t
                    if t is not None:
                        print(f"  => {bl_name} n={n}: {t:.3f}s")
                    else:
                        print(f"  => {bl_name} n={n}: FAILED")
                except Exception as e:
                    print(f"  => {bl_name} n={n}: EXCEPTION: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results[dataset][n][bl_name] = None

            # Print summary for this n
            parts = []
            for bl_name, _ in BASELINES:
                v = all_results[dataset][n].get(bl_name)
                if isinstance(v, (int, float)):
                    parts.append(f"{bl_name}={v:.3f}s")
                else:
                    parts.append(f"{bl_name}=FAIL")
            print(f"  => n={n}: " + ", ".join(parts))

    # Save results
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Print summary table
    print(f"\n{'='*70}")
    print("  SUMMARY (seconds, min of 3 runs)")
    print(f"{'='*70}")
    bl_names = [b[0] for b in BASELINES]
    for dataset in ["slice", "sst-2", "imagenet", "cifar-10"]:
        print(f"\n{dataset}:")
        header = f"  {'n':<8}"
        for bl in bl_names:
            header += f" {bl:<12}"
        print(header)
        print("  " + "-" * (8 + 12 * len(bl_names)))
        for n in N_VALUES:
            row = f"  {n:<8}"
            for bl in bl_names:
                v = all_results[dataset][n].get(bl, "FAIL")
                if isinstance(v, (int, float)):
                    row += f" {v:<12.3f}"
                else:
                    row += f" {'FAIL':<12}"
            print(row)

if __name__ == "__main__":
    main()
