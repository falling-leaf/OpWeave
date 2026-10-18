"""Load workers and timing helpers for the Table 3 scheduling experiment.

The two load models are the ones that produced
paper_workspace/MDB_Agent/tables/scheduling_load.tex. CPU phases overcommit
CPUs 0-19 (20/60/120 busy-wait workers). GPU phases run CUDA matmul workers
on GPU 0 (2x2GB, 3x2GB, 8x1.5GB with a forced sync).
"""
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time

ALLOWED_CPUS = list(range(20))
ALLOWED_CPU_SET = set(ALLOWED_CPUS)
ALLOWED_CPU_MASK = ",".join(str(c) for c in ALLOWED_CPUS)

DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "user": "why",
    "dbname": "postgres",
    "password": "123456",
}

CPU_PHASES = [
    {"name": "low", "n_workers": 20},
    {"name": "medium", "n_workers": 60},
    {"name": "high", "n_workers": 120},
]

GPU_PHASES = [
    {"name": "low", "n_workers": 2, "mem_gb": 2.0, "oom_style": False},
    {"name": "medium", "n_workers": 3, "mem_gb": 2.0, "oom_style": False},
    {"name": "high", "n_workers": 8, "mem_gb": 1.5, "oom_style": True},
]

_NEW_GPU_WORKER = """\
import torch, time, sys, os
os.sched_setaffinity(0, [0])
mem_gb = float(sys.argv[1]); stop_file = sys.argv[2]
n = min(int((mem_gb * 1024**3 / 4) ** 0.5), 8192)
a = torch.randn(n, n, device='cuda:0', dtype=torch.float32)
b = torch.randn(n, n, device='cuda:0', dtype=torch.float32)
c = torch.zeros(n, n, device='cuda:0', dtype=torch.float32)
while not os.path.exists(stop_file):
    with torch.no_grad():
        c.copy_(torch.matmul(a, b))
    time.sleep(0.05)
"""

_OOM_GPU_WORKER = """\
import torch, time, sys, os
os.sched_setaffinity(0, [0])
mem_gb = float(sys.argv[1]); stop_file = sys.argv[2]
n = min(int((mem_gb * 1024**3 / 4) ** 0.5), 8192)
a = torch.randn(n, n, device='cuda:0', dtype=torch.float32)
b = torch.randn(n, n, device='cuda:0', dtype=torch.float32)
while not os.path.exists(stop_file):
    with torch.no_grad():
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    time.sleep(0.001)
"""


def pin_this_process():
    os.sched_setaffinity(0, ALLOWED_CPUS)


def spin_worker_on_set(stop_flag):
    os.sched_setaffinity(0, ALLOWED_CPUS)
    while not stop_flag.value:
        pass


class CpuLoad:
    """Busy-wait workers spread across CPUs 0-19. 20/60/120 is 1:1, 3:1, 6:1."""

    def __init__(self):
        self.ctx = multiprocessing.get_context("fork")
        self.stop_flag = None
        self.procs = []

    def start(self, n_workers):
        self.stop_flag = self.ctx.Value("b", False)
        self.procs = []
        for _ in range(n_workers):
            proc = self.ctx.Process(target=spin_worker_on_set, args=(self.stop_flag,))
            proc.start()
            self.procs.append(proc)

    def stop(self):
        if self.stop_flag is not None:
            self.stop_flag.value = True
        for proc in self.procs:
            try:
                if proc.is_alive():
                    proc.kill()
            except Exception:
                pass
        for proc in self.procs:
            try:
                proc.join(timeout=1)
            except Exception:
                pass
        self.procs.clear()


class CpuMonitor:
    def __init__(self):
        self.prev_stats = {}
        self.initialized = False

    def read_stats(self):
        stats = {}
        with open("/proc/stat") as handle:
            for line in handle:
                if line.startswith("cpu") and not line.startswith("cpu "):
                    parts = line.strip().split()
                    idx = int(parts[0][3:])
                    if idx in ALLOWED_CPU_SET:
                        vals = [int(v) for v in parts[1:]]
                        active = vals[0] + vals[1] + vals[2] + vals[5] + vals[6]
                        stats[idx] = (active, sum(vals))
        return stats

    def warmup(self):
        self.prev_stats = self.read_stats()
        self.initialized = True

    def get_usage(self):
        curr = self.read_stats()
        if not self.initialized:
            self.prev_stats = curr
            self.initialized = True
            return 0.0
        total = 0.0
        count = 0
        for idx in ALLOWED_CPU_SET:
            if idx in self.prev_stats and idx in curr:
                active = curr[idx][0] - self.prev_stats[idx][0]
                span = curr[idx][1] - self.prev_stats[idx][1]
                if span > 0:
                    total += active / span * 100.0
                    count += 1
        self.prev_stats = curr
        return round(total / count, 1) if count else 0.0


class GpuLoad:
    """CUDA matmul workers on GPU 0. oom_style forces a sync and allocates every step."""

    def __init__(self):
        self.procs = []
        self.stop_file = None
        self.script_file = None

    def start(self, n_workers, mem_gb, oom_style=False):
        self.stop_file = tempfile.mktemp(suffix=".stop")
        script = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        script.write(_OOM_GPU_WORKER if oom_style else _NEW_GPU_WORKER)
        script.flush()
        script.close()
        self.script_file = script.name
        self.procs = []
        for _ in range(n_workers):
            proc = subprocess.Popen(
                ["taskset", "-c", ALLOWED_CPU_MASK, sys.executable, self.script_file,
                 str(mem_gb), self.stop_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.procs.append(proc)

    def stop(self):
        if self.stop_file:
            try:
                with open(self.stop_file, "w") as handle:
                    handle.write("stop")
            except Exception:
                pass
        time.sleep(2)
        for proc in self.procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self.procs.clear()
        for path in (self.stop_file, self.script_file):
            try:
                if path:
                    os.unlink(path)
            except Exception:
                pass


def gpu_util():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=0",
             "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        parts = [part.strip() for part in result.stdout.strip().split(",")]
        return float(parts[0]), float(parts[1])
    except Exception:
        return 0.0, 0.0


def cooldown(seconds, monitor):
    """Wait between phases. monitor is a CpuMonitor or None (GPU idle via nvidia-smi)."""
    print(f"\n  Cooling down for {seconds}s...")
    started = time.time()
    while time.time() - started < seconds:
        remaining = seconds - (time.time() - started)
        mins, secs = divmod(int(remaining), 60)
        if monitor is None:
            util, _ = gpu_util()
            print(f"    {mins:02d}:{secs:02d} remaining - GPU util: {util:.0f}%")
        else:
            print(f"    {mins:02d}:{secs:02d} remaining - idle CPU: {monitor.get_usage():.1f}%")
        time.sleep(min(10, max(0.0, remaining)))


def query_status(error):
    text = str(error).lower()
    if "timeout" in text or "canceling" in text:
        return "timeout"
    return "error"


def pin_backend(cur):
    cur.execute("SELECT pg_backend_pid()")
    backend_pid = cur.fetchone()[0]
    subprocess.run(
        ["taskset", "-p", "-c", ALLOWED_CPU_MASK, str(backend_pid)],
        capture_output=True, timeout=5,
    )
