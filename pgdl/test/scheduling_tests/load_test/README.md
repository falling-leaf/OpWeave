# Table 3 load test

One entry point for the scheduling table in
`paper_workspace/MDB_Agent/tables/scheduling_load.tex`. It replaces the nine
separate scripts that used to live in `scheduling_tests/` (now under
`recycle/scheduling_tests_table3/`). Nothing else in `scheduling_tests/` is
part of this table: slice, the scheduling-decision sweep, fault tolerance,
the all-GPU HotpotQA variants, and `optimization_ablation/` stay where they
are.

## What each cell measures

Reported latency is the minimum of the timed runs after one discarded warmup.
DBAgent and MorphingDB time the whole query, from connect to fetch. Hermes
times data loading plus inference, with the model loaded before the timer,
and the table adds a generation cost measured once with no load.

| cell | n | load | timed runs |
|---|---:|---|---:|
| dbagent_year_predict | 10000 | CPU overcommit, 20/60/120 workers on CPUs 0-19 | 3 |
| morphingdb_year_predict | 10000 | same | 3 |
| hermes_year_predict | 10000 | same | 3 |
| dbagent_cifar10 | 1000 | GPU 0 matmul, 2x2GB / 3x2GB / 8x1.5GB (high forces a sync) | 3 |
| morphingdb_cifar10 | 1000 | same | 3 |
| hermes_cifar10 | 1000 | CPU overcommit, not GPU load | 3 |
| dbagent_hotpotqa | 20 | GPU matmul, same as cifar10 | 1 |
| morphingdb_hotpotqa | 20 | GPU matmul, same as cifar10 | 1 |
| hermes_hotpotqa | 5 | CPU overcommit | 1 |

The HotpotQA header in the table says n=20. The DBAgent and MorphingDB cells
are n=20. The Hermes execution numbers in the paper come from the n=5 CPU
run, so this cell keeps n=5.

Hermes generation costs printed in the paper, used with `--skip-generation`:

| workload | ms |
|---|---:|
| year_predict | 24996.5 |
| cifar10 | 227541.3 |
| hotpotqa | 208110.0 |

## Before running

In `pgdl/src/pgdl/model_agent.cpp` set `ADD_OPTIMIZATION_LOGIC` to true and
leave `ALWAYS_TRAINING` false, then rebuild and install `pgdl.so` and restart
the server. Run one cell at a time. Every cell pins the test process, the
workers, and the PostgreSQL backend to CPUs 0-19, and the GPU cells also fill
GPU 0.

## Running

```bash
cd pgdl/test
python3 scheduling_tests/load_test/run_table3.py --list
python3 scheduling_tests/load_test/run_table3.py --cell dbagent_year_predict
python3 scheduling_tests/load_test/run_table3.py --workload year_predict --skip-generation
python3 scheduling_tests/load_test/run_table3.py --dry-run
```

`--skip-generation` reuses the three published generation costs.
`--generation-ms` overrides all of them with one number. Without either flag
the runner calls `hermes -z` three times per workload, under no load, and
keeps the minimum. Results go to
`pgdl/test/results/scheduling/load_test/table3_<timestamp>.json`, and
the run ends with each phase against the published number.

Published execution times, in seconds, low / med / high:

| system | year_predict | cifar10 | hotpotqa |
|---|---|---|---|
| DBAgent | 0.764 / 1.092 / 3.504 | 28.2 / 31.3 / 39.7 | 92.0 / 149.9 / 272.5 |
| MorphingDB | 1.079 / 1.892 / 4.918 | 31.8 / 35.6 / 48.0 | 168.6 / 268.7 / 345.1 |
| Hermes (exec only) | 0.233 / 0.239 / 0.982 | 62.9 / 93.9 / 200.6 | 148.9 / 298.3 / 610.0 |

A rerun will not hit these to the millisecond. The load is an oversubscribed
CPU or a shared GPU, so expect the same ordering and the same rough gaps.
`--rows` and `--repetitions` override the table's n and run count for a short
check; drop them for a real run.
