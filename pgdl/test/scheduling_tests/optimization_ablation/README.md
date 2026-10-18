# Optimization ablation

Reproduces the ablation of the device-selection logic reported in
`paper_workspace/MDB_Agent/tables/scheduling_ablation.tex`.

The two conditions differ only in the compile-time macro
`ADD_OPTIMIZATION_LOGIC` in `pgdl/src/pgdl/model_agent.cpp`. When it is false,
`ScheduleAgent` skips `OptimizationAgent` and every series query stays on CPU.
When it is true, the device is chosen once per query by counting the load
processes: CPU inference runs on CPU 0 and a GPU call is driven from CPU 20, so
a worker pinned to one core slows exactly one of the two paths. The query moves
to GPU 0 once the count reaches 3. `ALWAYS_TRAINING` stays false.

| workload | n | low | medium | high |
|---|---:|---|---|---|
| year_predict | 10000 | 1 spin on CPU 0 | 2 spins on CPU 0 | 4 spins on CPU 0, plus 1 wake worker on CPU 20 |
| slice | 10000 | same | same | same |
| swarm | 1461 | 1 spin at 5% duty | 1 spin at 25% duty | same as above |

A spin worker is a busy wait. A wake worker is two threads on one core waking
each other through a futex, which is what stalls the GPU driver thread; a busy
wait never sleeps, so the driver keeps the core and the GPU path does not slow
down. The decision counts processes named `spin_load` and `wake_load`, so those
binary names must not change.

Each cell is the minimum of 5 timed runs after 3 discarded warmups. The backend
is pinned to CPUs 0-19.

## Running

Do not run this while another agent is using this PostgreSQL instance. The
install restarts the `pg_base` server with `pg_ctl restart -m fast`, which
drops every backend.

```bash
bash run_experiment.sh            # build, then measure all three workloads
bash run_experiment.sh --dry-run  # print the steps, change nothing
```

The full run takes roughly an hour. Results are written as JSON under
`pgdl/test/results/scheduling/optimization_ablation/`.

The steps separately:

```bash
gcc -O2 -o load/spin_load load/spin_load.c
gcc -O2 -pthread -o load/wake_load load/wake_load.c
python3 prepare_binary.py --which both
python3 install_and_run.py --optimization on  --workloads year_predict,slice,swarm --cooldown 0
python3 install_and_run.py --optimization off --workloads year_predict,slice,swarm --cooldown 0
```

`prepare_binary.py` rewrites the `ADD_OPTIMIZATION_LOGIC` line for each build
and restores `model_agent.cpp` before it exits, including when the build fails.
An exclusive lock covers the edit. Build directories are
`optimization_ablation/build/{on,off}`, not `pgdl/build`.

`install_and_run.py` copies the chosen binary to
`pg_base/lib/postgresql/pgdl.so` (`$libdir/pgdl`). The previous file is copied
once to `pgdl.so.bak.optimization_ablation` and is never deleted. To install
without restarting, pass `--skip-restart` and restart the server yourself, then
call `run_ablation.py` directly.

## Layout

| file | role |
|---|---|
| `run_experiment.sh` | builds everything and measures both conditions |
| `prepare_binary.py` | builds the on and off binaries, restoring the source |
| `install_and_run.py` | installs one binary, restarts PostgreSQL, runs the workloads |
| `run_ablation.py` | one condition, one workload, three load phases |
| `workloads.py` | queries and the per-dataset load phases |
| `load_harness.py` | CPU spin, wake, and GPU matmul workers |
| `load/` | C sources for `spin_load` and `wake_load` |
| `fill_table_row.py` | renders a tex row from the saved JSON |
