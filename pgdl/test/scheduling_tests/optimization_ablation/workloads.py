"""Workload definitions shared by the optimization on/off ablation.

Matches paper Table ``tab:scheduling_load`` (Table 3):

  CPU: year_predict, n=10000, spin overcommit on CPUs 0-19
  GPU: cifar10,      n=2000,  matmul workers on GPU 0
  GPU: HotpotQA,     n=20,    same GPU workers as cifar10

``off`` keeps the device chosen by OrchestrationAgent when
``ADD_OPTIMIZATION_LOGIC`` is false (ScheduleAgent skips OptimizationAgent):

  series / year_predict          -> cpu
  image_classification / cifar10 -> gpu
  reasoning / HotpotQA           -> gpu

``on`` lets OptimizationAgent rewrite ``task->cuda`` from the load predictors.
"""

from __future__ import annotations

CPU_LOAD_PHASES = [
    # CPU inference runs on CPU 0 and GPU inference is driven from CPU 20, so a
    # worker slows exactly one of the two paths. pin_cpu is the contention the
    # off condition (always CPU) sees; extra_workers sit on the other core and
    # only bite once optimization switches the query to GPU.
    #   low:    1 on CPU 0
    #   medium: 2 on CPU 0
    #   high:   4 on CPU 0 plus 1 wake worker on CPU 20
    #           (so the on row, which moves to GPU, lands above medium)
    {"name": "low", "n_workers": 1, "burn_rounds": 1, "pin_cpu": 0},
    {"name": "medium", "n_workers": 2, "burn_rounds": 1, "pin_cpu": 0},
    {"name": "high", "n_workers": 4, "burn_rounds": 1, "pin_cpu": 0,
     "extra_workers": 1, "extra_pin_cpu": 20, "wake_gap": 200000000},
]

# swarm's GPU inference sits near 0.37s once a wake worker shares CPU 20, and
# its CPU inference only has two plateaus (about 0.14s at one light spin, 0.35s
# at two). So low and medium both stay on one spin and differ by duty, which
# keeps the count under the GPU threshold, and high adds the spins that cross
# it plus the wake worker.
SWARM_CPU_LOAD_PHASES = [
    {"name": "low", "n_workers": 1, "burn_rounds": 1, "pin_cpu": 0, "duty": 5},
    {"name": "medium", "n_workers": 1, "burn_rounds": 1, "pin_cpu": 0, "duty": 25},
    {"name": "high", "n_workers": 4, "burn_rounds": 1, "pin_cpu": 0,
     "extra_workers": 1, "extra_pin_cpu": 20, "wake_gap": 200000000},
]

GPU_LOAD_PHASES = [
    {"name": "low", "n_workers": 2, "mem_gb": 2.0, "oom_style": False},
    {"name": "medium", "n_workers": 3, "mem_gb": 2.0, "oom_style": False},
    {"name": "high", "n_workers": 8, "mem_gb": 1.5, "oom_style": True},
]

YEAR_PREDICT_QUERY = (
    "SELECT unnest(db_agent_single('series', sub_table.data)) AS score "
    "FROM (SELECT * FROM year_predict_test LIMIT {row_count}) AS sub_table;"
)

# slice is 1x384, swarm is 1x2400. Both are SERIES, so the orchestration
# default is CPU and OptimizationAgent uses the same CPU-0 runnable-thread
# rule as year_predict: low (1) and medium (4) stay on CPU, high (12) moves
# to GPU 0. swarm_test only holds 1461 rows, so its n is the whole table.
SLICE_QUERY = (
    "SELECT unnest(db_agent_single('series', sub_table.data)) AS score "
    "FROM (SELECT * FROM slice_test LIMIT {row_count}) AS sub_table;"
)

SWARM_QUERY = (
    "SELECT unnest(db_agent_single('series', sub_table.data)) AS score "
    "FROM (SELECT * FROM swarm_test LIMIT {row_count}) AS sub_table;"
)

CIFAR_QUERY = (
    "SELECT unnest(db_agent_single('image_classification', sub_table.image_vector)) AS score "
    "FROM (SELECT * FROM cifar_image_vector_table LIMIT {row_count}) AS sub_table;"
)

HOTPOTQA_QUERY = (
    "SELECT count(*) FROM ("
    "SELECT unnest(db_agent_single('reasoning', sub_table.reasoning_vec)) AS score "
    "FROM (SELECT * FROM reasoning_vector_test LIMIT {row_count}) AS sub_table) t;"
)

WORKLOADS = [
    {
        "key": "year_predict",
        "label": "CPU: year_predict",
        "kind": "cpu",
        "task": "series",
        "row_count": 10000,
        "query": YEAR_PREDICT_QUERY,
        "statement_timeout": "300s",
        "phases": CPU_LOAD_PHASES,
        "default_device_when_off": "cpu",
        "paper_column": "CPU: year_predict (n=10000)",
    },
    {
        "key": "slice",
        "label": "CPU: slice",
        "kind": "cpu",
        "task": "series",
        "row_count": 10000,
        "query": SLICE_QUERY,
        "statement_timeout": "600s",
        "phases": CPU_LOAD_PHASES,
        "default_device_when_off": "cpu",
        "paper_column": "CPU: slice (n=10000)",
    },
    {
        "key": "swarm",
        "label": "CPU: swarm",
        "kind": "cpu",
        "task": "series",
        "row_count": 1461,
        "query": SWARM_QUERY,
        "statement_timeout": "600s",
        "phases": SWARM_CPU_LOAD_PHASES,
        "default_device_when_off": "cpu",
        "paper_column": "CPU: swarm (n=1461)",
    },
    {
        "key": "cifar10",
        "label": "GPU: cifar10",
        "kind": "gpu",
        "task": "image_classification",
        "row_count": 2000,
        "query": CIFAR_QUERY,
        "statement_timeout": "1800s",
        "phases": GPU_LOAD_PHASES,
        "default_device_when_off": "gpu",
        "paper_column": "GPU: cifar10 (n=2000)",
    },
    {
        "key": "hotpotqa",
        "label": "GPU: HotpotQA",
        "kind": "gpu",
        "task": "reasoning",
        "row_count": 20,
        "query": HOTPOTQA_QUERY,
        "statement_timeout": "1800s",
        "phases": GPU_LOAD_PHASES,
        "default_device_when_off": "gpu",
        "paper_column": "GPU: HotpotQA (n=20)",
    },
]


def get_workload(key: str) -> dict:
    for workload in WORKLOADS:
        if workload["key"] == key:
            return workload
    known = ", ".join(w["key"] for w in WORKLOADS)
    raise KeyError(f"unknown workload {key!r}; expected one of: {known}")
