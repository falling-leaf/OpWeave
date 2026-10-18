#!/usr/bin/env python3
"""
Hermes Single-Query Agent Test — Code Generation + Execution Timing
====================================================================
Tests Hermes as an agent: LLM generates Python code, then hermes executes it.
Measures separately:
  - codegen_s:  time from prompt submission to code generation finish (LLM latency)
  - exec_s:     time for actual inference/pipeline execution (parsed from hermes stdout)
  - total_s:    wall-clock time (codegen_s + exec_s + overhead)

Datasets: slice, sst-2, imagenet, cifar-10  (n in {2000, 4000, 6000, 8000, 10000})
          HotpotQA, musique                  (n in {1, 5, 10, 20})

Methodology:
  - hermes is asked to print EXEC_START_MS=<ms> and EXEC_END_MS=<ms> around inference
  - codegen_s = time from subprocess start to EXEC_START_MS being printed
    (approximated from total - exec_s since EXEC_START_MS parsing is post-hoc)
  - exec_s = (EXEC_END_MS - EXEC_START_MS) / 1000.0
  - total_s = wall-clock from subprocess.Popen to completion
  - 1 warm-up + 3 timed runs, report min
"""

import os, sys, time, json, re, subprocess, gc
from datetime import datetime

HERMES_CLI = "/home/why/.local/bin/hermes"
BASELINE_DIR = "/home/why/baseline"
DATA_DIR = os.path.join(BASELINE_DIR, "data")
MODELS_DIR = os.path.join(BASELINE_DIR, "models")
TOOLS_DIR = "/home/why/dbagent/pgdl/test/tools"
from tools.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir

RESULTS_DIR = ensure_dir(GENERAL_RESULTS_DIR)
TIMEOUT_S = 600   # 10 minutes per run max

N_VALUES_DL_IMAGE = [2000, 4000, 6000, 8000, 10000]
N_VALUES_REASONING = [1, 5, 10, 20]

# ── Prompt templates ─────────────────────────────────────────────────────────
# Each prompt instructs hermes to:
#  1. Load data/model
#  2. Print EXEC_START_MS=<time.perf_counter()*1000>
#  3. Run inference
#  4. Print EXEC_END_MS=<time.perf_counter()*1000>
# This lets us extract exec time from stdout.

def make_prompt_slice(n):
    return (
        f"Write and run Python code that does the following steps in order:\n"
        f"1. import time, numpy as np, pandas as pd, torch\n"
        f"2. model = torch.jit.load('{MODELS_DIR}/slice.pt', map_location='cpu'); model.eval()\n"
        f"3. df = pd.read_csv('{DATA_DIR}/series/slice/slice_localization_data.csv', nrows={n}); "
        f"data = df.iloc[:, 1:-1].values.astype('float32')\n"
        f"4. IMPORTANT: print the line: EXEC_START_MS=<t0> where t0 = time.perf_counter()*1000.0\n"
        f"5. Run batched inference: batch_size=32, torch.no_grad(), iterate over data\n"
        f"6. IMPORTANT: print the line: EXEC_END_MS=<t1> where t1 = time.perf_counter()*1000.0\n"
        f"7. Print output shape.\n"
        f"The EXEC_START_MS and EXEC_END_MS lines must be printed to stdout with no extra text on the same line."
    )

def make_prompt_sst2(n):
    return (
        f"Write and run Python code that does the following steps in order:\n"
        f"1. import time, numpy as np, pandas as pd, torch\n"
        f"2. model = torch.jit.load('{MODELS_DIR}/traced_albert_vec.pt', map_location='cuda'); model.eval()\n"
        f"3. Load {n} rows from '{DATA_DIR}/text/sst2/data/train.tsv' (tab-sep, header, 'sentence' column)\n"
        f"4. Tokenize with a simple word tokenizer: max_len=128, pad with 0, input_ids/attention_mask/"
        f"token_type_ids(zeros)/position_ids(0..127), all torch.long on CUDA\n"
        f"5. IMPORTANT: print the line: EXEC_START_MS=<t0> where t0 = time.perf_counter()*1000.0\n"
        f"6. Run batched inference: batch_size=32, torch.no_grad()\n"
        f"7. IMPORTANT: print the line: EXEC_END_MS=<t1> where t1 = time.perf_counter()*1000.0\n"
        f"8. Print output shape.\n"
        f"The EXEC_START_MS and EXEC_END_MS lines must each be on their own line with no extra text."
    )

def make_prompt_imagenet(n):
    return (
        f"Write and run Python code that does the following steps in order:\n"
        f"1. import time, cv2, numpy as np, torch, glob\n"
        f"2. model = torch.jit.load('{MODELS_DIR}/resnet18_imagenet.pt', map_location='cuda'); model.eval()\n"
        f"3. paths = sorted(glob.glob('{DATA_DIR}/image/image-net/data/*.JPEG'))[:{n}]\n"
        f"4. Preprocess each image: cv2.imread -> BGR2RGB -> resize(224,224) -> float32/255 -> "
        f"normalize mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] -> CHW tensor\n"
        f"5. IMPORTANT: print the line: EXEC_START_MS=<t0> where t0 = time.perf_counter()*1000.0\n"
        f"6. Run batched inference: batch_size=32, torch.no_grad(), move batch to CUDA\n"
        f"7. IMPORTANT: print the line: EXEC_END_MS=<t1> where t1 = time.perf_counter()*1000.0\n"
        f"8. Print output shape.\n"
        f"The EXEC_START_MS and EXEC_END_MS lines must each be on their own line with no extra text."
    )

def make_prompt_cifar10(n):
    classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    per = (n // 10) + 1
    cls_str = str(classes)
    return (
        f"Write and run Python code that does the following steps in order:\n"
        f"1. import time, cv2, numpy as np, torch, glob, os\n"
        f"2. model = torch.jit.load('{MODELS_DIR}/googlenet_cifar10.pt', map_location='cuda'); model.eval()\n"
        f"3. Collect images: for each class in {cls_str}, "
        f"collect up to {per} .png files from '{DATA_DIR}/image/cifar10/train/<class>/' and "
        f"'{DATA_DIR}/image/cifar10/test/<class>/' (sorted). Take first {n} total.\n"
        f"4. Preprocess each image: cv2.imread -> BGR2RGB -> resize(224,224) -> float32/255 -> "
        f"normalize mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] -> CHW tensor\n"
        f"5. IMPORTANT: print the line: EXEC_START_MS=<t0> where t0 = time.perf_counter()*1000.0\n"
        f"6. Run batched inference: batch_size=32, torch.no_grad(), move batch to CUDA\n"
        f"7. IMPORTANT: print the line: EXEC_END_MS=<t1> where t1 = time.perf_counter()*1000.0\n"
        f"8. Print output shape.\n"
        f"The EXEC_START_MS and EXEC_END_MS lines must each be on their own line with no extra text."
    )

def make_prompt_hotpotqa(n):
    return (
        f"Write and run Python code that does ALL of these steps completely:\n"
        f"1. import time, numpy as np, torch, subprocess, psycopg2\n"
        f"2. Connect to PostgreSQL: host=localhost port=5432 user=why password=123456 dbname=postgres\n"
        f"3. Load {n} rows: SELECT get_mvec_data(reasoning_vec) FROM reasoning_vector_test ORDER BY id LIMIT {n}\n"
        f"   Each row is a flat float array -> reshape to (4,128). Call this input_vecs.\n"
        f"4. Load {n*2} rows: SELECT get_mvec_data(text_vec) FROM reasoning_vector_step2 ORDER BY id LIMIT {n*2}\n"
        f"   Each row is a flat float array -> reshape to (4,128). Call this step2_vecs.\n"
        f"5. Load cross_encoder: torch.jit.load('/home/why/dbagent/pgdl/test/morphingdb_test/models/cross_encoder.pt', map_location='cpu'); eval()\n"
        f"6. Load deberta_reader: torch.jit.load('/home/why/dbagent/pgdl/test/morphingdb_test/models/deberta_reader.pt', map_location='cpu'); eval()\n"
        f"7. IMPORTANT: print the line: EXEC_START_MS=<t0> where t0 = time.perf_counter()*1000.0\n"
        f"8. Stage 1 (cross_encoder): for all {n} input_vecs, stack first 2 rows of each -> batch shape ({n},2,128) -> run cross_encoder -> scores\n"
        f"9. Stage 2 (deberta_reader): for each sample i in 0..{n-1}:\n"
        f"   - take step2_vecs[i*2] and step2_vecs[i*2+1], stack first 2 rows each -> shape (2,2,128)\n"
        f"   - run deberta_reader -> output\n"
        f"10. Stage 3 (reader_decoder): for each sample i, extract start/end logits and input_ids from step2_vecs,\n"
        f"    build ids_string as 'id1,id2,...;id1,id2,...', then call:\n"
        f"    subprocess.run(['uv','run','{TOOLS_DIR}/reader_decoder.py','--ids',ids_string], capture_output=True, text=True, cwd='{TOOLS_DIR}', timeout=60)\n"
        f"    Collect decoded text from stdout. YOU MUST complete this for ALL {n} samples.\n"
        f"11. IMPORTANT: print the line: EXEC_END_MS=<t1> where t1 = time.perf_counter()*1000.0\n"
        f"12. Print how many answers were decoded.\n"
        f"The EXEC_START_MS and EXEC_END_MS lines must each be on their own line. Do not skip any stage."
    )

def make_prompt_musique(n):
    return (
        f"Write and run Python code that does ALL of these steps completely:\n"
        f"1. import time, numpy as np, torch, subprocess, psycopg2\n"
        f"2. Connect to PostgreSQL: host=localhost port=5432 user=why password=123456 dbname=postgres\n"
        f"3. Load {n} rows: SELECT get_mvec_data(reasoning_vec) FROM musique_vector_test ORDER BY id LIMIT {n}\n"
        f"   Each row is a flat float array -> reshape to (4,128). Call this input_vecs.\n"
        f"4. Load {n*2} rows: SELECT get_mvec_data(text_vec) FROM musique_vector_step2 ORDER BY id LIMIT {n*2}\n"
        f"   Each row is a flat float array -> reshape to (4,128). Call this step2_vecs.\n"
        f"5. Load cross_encoder: torch.jit.load('/home/why/dbagent/pgdl/test/morphingdb_test/models/cross_encoder.pt', map_location='cpu'); eval()\n"
        f"6. Load deberta_reader: torch.jit.load('/home/why/dbagent/pgdl/test/morphingdb_test/models/deberta_reader.pt', map_location='cpu'); eval()\n"
        f"7. IMPORTANT: print the line: EXEC_START_MS=<t0> where t0 = time.perf_counter()*1000.0\n"
        f"8. Stage 1 (cross_encoder): for all {n} input_vecs, stack first 2 rows of each -> batch shape ({n},2,128) -> run cross_encoder -> scores\n"
        f"9. Stage 2 (deberta_reader): for each sample i in 0..{n-1}:\n"
        f"   - take step2_vecs[i*2] and step2_vecs[i*2+1], stack first 2 rows each -> shape (2,2,128)\n"
        f"   - run deberta_reader -> output\n"
        f"10. Stage 3 (reader_decoder): for each sample i, extract start/end logits and input_ids from step2_vecs,\n"
        f"    build ids_string as 'id1,id2,...;id1,id2,...', then call:\n"
        f"    subprocess.run(['uv','run','{TOOLS_DIR}/reader_decoder.py','--ids',ids_string], capture_output=True, text=True, cwd='{TOOLS_DIR}', timeout=60)\n"
        f"    Collect decoded text from stdout. YOU MUST complete this for ALL {n} samples.\n"
        f"11. IMPORTANT: print the line: EXEC_END_MS=<t1> where t1 = time.perf_counter()*1000.0\n"
        f"12. Print how many answers were decoded.\n"
        f"The EXEC_START_MS and EXEC_END_MS lines must each be on their own line. Do not skip any stage."
    )

PROMPT_MAKERS = {
    "slice":    make_prompt_slice,
    "sst-2":    make_prompt_sst2,
    "imagenet": make_prompt_imagenet,
    "cifar-10": make_prompt_cifar10,
    "HotpotQA": make_prompt_hotpotqa,
    "musique":  make_prompt_musique,
}

# ── Runner ───────────────────────────────────────────────────────────────────
def run_hermes_once(dataset, n):
    """
    Run one hermes agent call. Returns dict with:
      total_s, exec_s, codegen_s, success, stdout (truncated)
    """
    prompt = PROMPT_MAKERS[dataset](n)
    gc.collect()

    t_start = time.perf_counter()
    try:
        result = subprocess.run(
            [HERMES_CLI, "-z", prompt, "--yolo"],
            capture_output=True, text=True,
            timeout=TIMEOUT_S,
            cwd=BASELINE_DIR,
        )
        t_total = time.perf_counter() - t_start
    except subprocess.TimeoutExpired:
        return {"total_s": None, "exec_s": None, "codegen_s": None,
                "success": False, "error": "TIMEOUT"}
    except Exception as e:
        return {"total_s": None, "exec_s": None, "codegen_s": None,
                "success": False, "error": str(e)}

    stdout = result.stdout + result.stderr
    success = result.returncode == 0

    # Parse EXEC_START_MS and EXEC_END_MS from stdout.
    # Hermes may format the value with '=', ': ', backticks, or bullet points.
    exec_s = None
    m_start = re.search(r'EXEC_START_MS\s*[=:]\s*`?([0-9.]+)`?', stdout)
    m_end   = re.search(r'EXEC_END_MS\s*[=:]\s*`?([0-9.]+)`?', stdout)
    if m_start and m_end:
        exec_ms = float(m_end.group(1)) - float(m_start.group(1))
        exec_s  = max(exec_ms / 1000.0, 0.0)

    codegen_s = (t_total - exec_s) if exec_s is not None else None

    return {
        "total_s":   round(t_total, 4),
        "exec_s":    round(exec_s, 4) if exec_s is not None else None,
        "codegen_s": round(codegen_s, 4) if codegen_s is not None else None,
        "success":   success,
        "returncode": result.returncode,
        "stdout_tail": stdout[-800:],
    }


def benchmark_one(dataset, n, num_runs=3):
    """1 warm-up + num_runs timed runs. Returns dict of min values."""
    print(f"    warm-up (n={n})...", end=" ", flush=True)
    warm = run_hermes_once(dataset, n)
    if warm["total_s"] is None:
        print(f"FAILED: {warm.get('error','?')}")
        return None
    print(f"total={warm['total_s']:.2f}s exec={warm['exec_s']}s codegen={warm['codegen_s']}s")

    runs = []
    for i in range(num_runs):
        print(f"    run {i+1}/{num_runs} (n={n})...", end=" ", flush=True)
        r = run_hermes_once(dataset, n)
        if r["total_s"] is None:
            print(f"FAILED: {r.get('error','?')}")
            continue
        runs.append(r)
        print(f"total={r['total_s']:.2f}s exec={r['exec_s']}s codegen={r['codegen_s']}s")

    if not runs:
        return None

    best = min(runs, key=lambda x: x["total_s"])
    print(f"    => n={n}: total={best['total_s']:.3f}s  exec={best['exec_s']}s  codegen={best['codegen_s']}s")
    return best


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"  HERMES SINGLE-QUERY AGENT TEST — {datetime.now().isoformat()}")
    print("=" * 70)
    print("  Measures: codegen (LLM) + execution (inference) + total")
    print("  1 warm-up + 3 timed runs, report min of timed runs")
    print("=" * 70)

    all_results = {}

    DATASET_N = {
        "slice":    N_VALUES_DL_IMAGE,
        "sst-2":    N_VALUES_DL_IMAGE,
        "imagenet": N_VALUES_DL_IMAGE,
        "cifar-10": N_VALUES_DL_IMAGE,
        "HotpotQA": N_VALUES_REASONING,
        "musique":  N_VALUES_REASONING,
    }

    for dataset in DATASET_N:
        all_results[dataset] = {}
        print(f"\n{'#'*60}")
        print(f"# DATASET: {dataset}")
        print(f"{'#'*60}")

        for n in DATASET_N[dataset]:
            print(f"\n  --- n={n} ---")
            result = benchmark_one(dataset, n)
            if result:
                all_results[dataset][str(n)] = {
                    "total_s":   result["total_s"],
                    "exec_s":    result["exec_s"],
                    "codegen_s": result["codegen_s"],
                }
            else:
                all_results[dataset][str(n)] = {
                    "total_s": None, "exec_s": None, "codegen_s": None
                }

    # Save JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_json = os.path.join(RESULTS_DIR, f"hermes_single_test_{ts}.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Print summary table
    print(f"\n{'='*70}")
    print("  SUMMARY (seconds, min of 3 timed runs)")
    print(f"{'='*70}")
    for dataset, n_map in all_results.items():
        n_list = list(n_map.keys())
        print(f"\n{dataset}:")
        print(f"  {'n':<8} {'total_s':<12} {'exec_s':<12} {'codegen_s':<12}")
        print("  " + "-" * 44)
        for n_str in n_list:
            r = n_map[n_str]
            def fmt(v):
                return f"{v:.4f}" if v is not None else "FAIL"
            print(f"  {n_str:<8} {fmt(r['total_s']):<12} {fmt(r['exec_s']):<12} {fmt(r['codegen_s']):<12}")

    # Generate LaTeX table
    print(f"\n{'='*70}")
    print("  LATEX TABLE (hermes_single_overhead.tex)")
    print(f"{'='*70}")
    generate_latex(all_results, os.path.join(RESULTS_DIR, f"hermes_single_test_{ts}.tex"))


def generate_latex(results, outpath):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \caption{Hermes agent end-to-end latency breakdown: code generation (LLM) vs.\ execution (inference) per dataset and scale. Times in seconds.}")
    lines.append(r"  \label{tab:hermes_single_overhead}")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(r"  \begin{tabular}{llrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    Dataset & $n$ & Total (s) & Exec (s) & CodeGen (s) \\")
    lines.append(r"    \midrule")

    def fmt(v):
        if v is None:
            return "--"
        return f"{v:.3f}"

    dl_image = ["slice", "sst-2", "imagenet", "cifar-10"]
    reasoning = ["HotpotQA", "musique"]

    for group_idx, datasets in enumerate([dl_image, reasoning]):
        if group_idx > 0:
            lines.append(r"    \midrule")
        for ds_idx, ds in enumerate(datasets):
            if ds not in results:
                continue
            n_map = results[ds]
            n_list = sorted(n_map.keys(), key=lambda x: int(x))
            nrows = len(n_list)
            for i, n_str in enumerate(n_list):
                r = n_map[n_str]
                if i == 0:
                    lines.append(f"    \\multirow{{{nrows}}}{{*}}{{{ds}}} & {n_str} & {fmt(r['total_s'])} & {fmt(r['exec_s'])} & {fmt(r['codegen_s'])} \\\\")
                else:
                    lines.append(f"    & {n_str} & {fmt(r['total_s'])} & {fmt(r['exec_s'])} & {fmt(r['codegen_s'])} \\\\")
            if ds_idx < len(datasets) - 1:
                lines.append(r"    \cmidrule{1-5}")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    with open(outpath, "w") as f:
        f.write(latex)
    print(latex)
    print(f"\nLaTeX saved to {outpath}")


if __name__ == "__main__":
    main()
