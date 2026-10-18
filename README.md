# OpWeave

OpWeave is a database-native system for **stateful agentic workflow execution**, implemented as an
extension to **PostgreSQL 13.22**. It lifts agent workflows to first-class execution objects inside
the engine: relational, vector, model, and agent operations are organized into dependency-aware
workflow graphs, a stateful runtime coordinates them through shared execution state, and a cost- and
reuse-aware optimizer decides placement across CPUs and GPUs.

This repository contains the extension, the deployment scripts, and the test harness that produces
the experiments reported in *OpWeave: Native Execution of Stateful Agentic Workflows in Database
Systems*.

## Repository layout

| Path | Contents |
|---|---|
| `pgdl/src/pgdl/` | C++ engine: agent runtime, optimizer, load predictors, `mvec` vector type, model manager |
| `pgdl/src/external_process/` | Input/output handlers registered against models |
| `pgdl/sql/` | Extension SQL, versioned `0.1.0` through `1.3.0` |
| `pgdl/shell/` | Deployment scripts (build dependencies, install PostgreSQL, compile the extension) |
| `pgdl/test/` | Experiment harness (see [Test harness](#test-harness)) |
| `pgdl/visualization/` | Plotting scripts and the shipped figures |
| `pgdl/CMakeLists.txt`, `pgdl/pgdl.control` | Build and extension-package metadata |

Not tracked by git, and produced locally by the steps below: `pg_base/` (the PostgreSQL install),
`pgdl/third_party/` (dependencies), `pgdl/test/morphingdb_test/{data,models}/`, and
`pgdl/test/results/` (all run output).

## Prerequisites

* Linux host matching the paper configuration: Intel Xeon Silver 4210R (40 threads) and four
  NVIDIA RTX 3090 GPUs (24 GB each). The load and isolation tests pin explicit CPU ranges and a
  dedicated GPU, so fewer cores will change their numbers.
* CUDA toolkit at `/usr/local/cuda`.
* `sudo` (the install scripts run `make install` and `pg_ctl`).
* Python 3.12 with [`uv`](https://docs.astral.sh/uv/): `pgdl/test/pyproject.toml` and `uv.lock`
  pin the harness environment.
* A PostgreSQL 13.22 source tree extracted at `pgdl/third_party/postgresql-13.22`. The scripts do
  not fetch it; download and unpack it there before step 2.

**The scripts assume the checkout is at `/home/why/dbagent`.** Absolute paths appear throughout the
shell scripts, the harness, and two places in the C++ engine
(`pgdl/src/pgdl/model_agent.cpp`, which launches `pgdl/test/tools/reader_decoder.py` and
`pgdl/test/tools/llm_generate.py`). Deploying elsewhere means editing those paths *and* rebuilding
`pgdl.so`.

## Deployment

Run every command from the repository root.

### 1. Fetch and build the third-party dependencies

```bash
bash pgdl/shell/download_build_thirdparty.sh
```

Clones and builds SentencePiece, and downloads `libtorch` 2.0.0 (CPU), OpenCV 3.4.16, and
onnxruntime 1.18.1 into `pgdl/third_party/`. It does not fetch PostgreSQL, and its last two steps
copy CMake config files from a `pgdl/shell/config/` directory that is not part of this repository;
if those files are missing, run the script up to that point and add them by hand.

### 2. Build the PostgreSQL base

```bash
bash pgdl/shell/postgresql_install.sh
```

Configures, builds, and installs PostgreSQL 13.22 into `pg_base/`, then runs `initdb` on
`pg_base/data`.

### 3. Build and install the extension

```bash
bash pgdl/shell/compile.sh
```

Stops the server, rebuilds `pgdl.so` against `pg_base`, libtorch, and onnxruntime, installs it, and
restarts nothing. Re-run this after every change under `pgdl/src/`.

### 4. Start the server

```bash
bash pgdl/shell/run.sh
```

Restarts the server and drops into `psql` on `postgres`. The harness connects to
`localhost:5432`, database `postgres`, user `why`, password `123456`
(`pgdl/test/morphingdb_test/config.py`).

### 5. Models and data

The model weights (~23 GB) and datasets (~7.9 GB) are not in the repository. Fetch them and unzip
into the harness tree:

```bash
unzip models.zip -d pgdl/test/morphingdb_test/
unzip data.zip   -d pgdl/test/morphingdb_test/
```

Links are in `pgdl/README.md`. Paths the harness expects are
`pgdl/test/morphingdb_test/models/` and `pgdl/test/morphingdb_test/data/`.

### 6. Initialise the datasets and register the models

```bash
bash pgdl/test/run.sh
```

Creates `pgdl_agent`'s tables (`*_test`, `*_vector_*`, `reasoning_vector_step2`, …) and registers
the TorchScript models in `model_info`. This is a one-time step and takes a while for the image
datasets.

### 7. Verify the installation

```bash
cd pgdl/test
python3 general_tests/verify_supported_tests.py
```

Runs the smoke/regression set. Use `--list` to see which scripts it covers and `--list-long` for
the longer scripts it deliberately excludes.

## Running the experiments

All commands below assume `cd pgdl/test`. Every script writes JSON (and sometimes CSV) under
`pgdl/test/results/<category>/` — never next to the script.

### Reported in the paper

| Paper artifact | Script |
|---|---|
| End-to-end latency, OpWeave, six workloads, `n ∈ {2000,4000,6000,8000,10000}` (AI-native) and `n ∈ {1,5,10,20}` (agent-native) | `general_tests/dbagent_latency_new_scale.py` |
| End-to-end latency, five baselines (EvaDB, MorphingDB, NeurDB, GenDB, Hermes) | `general_tests/baseline_latency_new_scale.py`, `general_tests/benchmark_all_baselines.py` |
| Throughput (maximum safe concurrency), all systems | `throughput_tests/test_throughput_main.py` and the per-system scripts; run them together with `python3 run_throughput_tests.py` |
| Execution-time breakdown on 20 HotpotQA samples: inference / materialization / orchestration | `general_tests/test_agent_timing.py` (OpWeave), `general_tests/hermes_agent_breakdown.py` (Hermes); `general_tests/hotpotqa_agent_breakdown.py` records the per-agent timings from the SQL path |
| Cumulative latency of repeated queries in one session, up to 100 queries on `year_predict`, `slice`, `swarm` | `general_tests/db_session_reuse.py` (OpWeave and MorphingDB), `general_tests/hermes_session_reuse.py` (Hermes) |
| Latency under CPU and GPU contention, low / medium / high, on `year_predict`, `cifar10`, HotpotQA | `scheduling_tests/load_test/run_table3.py` |
| Device-selection ablation (`ADD_OPTIMIZATION_LOGIC` on vs. off) on three series workloads | `scheduling_tests/optimization_ablation/run_experiment.sh` |
| Sensitivity to data skew, 53,491 rows, ratios from 50:50 to 99:1 | `general_tests/test_skew.py` |
| Runtime resource usage (latency, CPU time, host memory, GPU memory) on HotpotQA and MuSiQue versus Hermes | `general_tests/test_overhead.py`, `general_tests/test_overhead_comparison.py`, `general_tests/test_hermes_overhead.py` |
| Appendix: explicit MorphingDB operator pipeline versus the single-call OpWeave interface | `morphingdb_test/run_pipeline.py`, `morphingdb_test/run_pipeline_20.py` |

### Workloads

Six workloads are reported: `slice` and `sst-2` (AI-native data/model inference), `imagenet` and
`cifar-10` (image classification), and HotpotQA and MuSiQue (multi-hop reasoning). The session and
scheduling experiments additionally use the series datasets `swarm` and `year_predict`. The harness
still implements the original eleven-workload scope; `swarm`, `year_predict`, `imdb`, `financial`,
and `stanford_dogs` remain runnable but are outside the reported subset.

Some experiments are sensitive to how the models were produced or to the machine state:

* The CIFAR-10 and ImageNet latency measurements run over **stored image vectors**, not path-based
  images.
* The scheduling and overhead experiments pin the process, the workers, and the PostgreSQL backend
  to CPUs 0–19 and reserve GPU 0 or GPU 2 respectively. Run them one at a time, with nothing else
  using the instance.
* `scheduling_tests/optimization_ablation/run_experiment.sh` restarts the server with a different
  `pgdl.so`; do not start it while another experiment is running.

### The optimization toggle

`ADD_OPTIMIZATION_LOGIC` in `pgdl/src/pgdl/model_agent.cpp` is compiled into `pgdl.so`.

* The **scheduling** experiments need it **on** — they install their own binaries with
  `optimization_ablation/prepare_binary.py`.
* Every other experiment needs it **off**, so that `ScheduleAgent` skips `OptimizationAgent` and
  each task keeps the device chosen during orchestration. `run_general_tests.py` and the general
  tests call `tools/general_extension.py:ensure_opt_off()`, which installs
  `optimization_ablation/bin/pgdl_opt_off.so` and restarts the server only when the installed binary
  differs.

If a general test reports unexpected placement, check which `pgdl.so` is installed before rerunning.

## Test harness

```
pgdl/test/
├── general_tests/          correctness, breakdown, overhead, latency
├── throughput_tests/       concurrency and throughput
├── scheduling_tests/       CPU/GPU load, under-load, and scheduling decisions
│   ├── load_test/          one entry point for the contention table
│   └── optimization_ablation/
├── init_tests/             dataset import and model registration
├── morphingdb_test/        the MorphingDB/OpWeave comparison pipelines, plus models and data
└── tools/                  shared helpers (result paths, binary install) and the model-side
                            scripts the C++ engine launches
```

The two scheduling subtrees carry their own READMEs describing their load models and expected
numbers: `scheduling_tests/load_test/README.md` and
`scheduling_tests/optimization_ablation/README.md`.

Category runners:

```bash
python3 run_init_tests.py
python3 run_general_tests.py
python3 run_throughput_tests.py
python3 run_load_tests.py
```

`run_general_tests.py`, `run_throughput_tests.py`, and `run_load_tests.py` accept `--list` and
print the scripts they would run instead of running them.

`tools/test_result_paths.py` resolves the output directories
(`results/general/`, `results/throughput/`, `results/scheduling/`, `results/init/`). Build result
paths through it rather than by hand. The whole `pgdl/test/results/` tree is run output and is
git-ignored; the pre-2026-07 collection is archived under `results/legacy/`.

## Baselines

EvaDB, MorphingDB, NeurDB, and GenDB run through their own installations and are driven by the
`benchmark_*` and `test_*` baseline scripts. Hermes runs through its external runtime, optionally
with the generation cost skipped (the scheduling runner can reuse the published generation costs
with `--skip-generation`). EvaDB appears only in the AI-native latency comparison: its throughput
and agent-native measurements are unavailable and are omitted rather than reported as zero.
