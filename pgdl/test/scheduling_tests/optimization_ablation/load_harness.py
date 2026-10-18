"""CPU and GPU contention helpers for the optimization ablation.

Copied in spirit from the existing scheduling tests so this package does not
import those modules (another agent may be executing them).
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import time

# 0-20, not 0-19: GPU inference pins its calling thread to CPU 20 inside
# infer_batch_internal, and a mask that stops at 19 silently drops that pin.
ALLOWED_CPUS = list(range(21))
ALLOWED_CPU_SET = set(ALLOWED_CPUS)
ALLOWED_CPU_MASK = ",".join(str(c) for c in ALLOWED_CPUS)

NEW_WORKER_SCRIPT = """\
import torch, time, sys, os
# Stay off CPU 0. The image fallback runs CPU inference there, and a worker
# pinned to the same core starves it (the oom-style loop never blocks).
os.sched_setaffinity(0, set(range(1, 20)))
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

OOM_WORKER_SCRIPT = """\
import torch, time, sys, os
os.sched_setaffinity(0, set(range(1, 20)))
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


def pin_current_process() -> None:
    os.sched_setaffinity(0, ALLOWED_CPUS)


def pin_backend(pid: int) -> None:
    subprocess.run(
        ["taskset", "-p", "-c", ALLOWED_CPU_MASK, str(pid)],
        capture_output=True,
        timeout=5,
    )


LOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "load")


def _wake_worker(stop_flag, pin_cpu: int, gap: int = 0) -> None:
    # Two threads pinned to one core, blocking on each other through a futex.
    # That keeps the core in a context switch instead of a busy wait, which is
    # what actually stalls the GPU driver thread. A busy wait does not: it
    # never sleeps, so the driver keeps the core. The binary name must stay
    # wake_load: the device decision counts processes by that name.
    proc = subprocess.Popen([os.path.join(LOAD_DIR, "wake_load"), str(pin_cpu), str(gap)])
    # _renice_normal(proc.pid)  # kept at the caller's priority for this run
    while not stop_flag.value and proc.poll() is None:
        time.sleep(0.2)
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()


def _renice_normal(pid: int) -> None:
    # The benchmark shell is niced to -10, and a forked worker inherits that,
    # which lets a 5% spin outrank the backend (nice -10 too, but the spin never
    # blocks while it burns). Pin every load worker to nice 0 so the duty cycle
    # is the only contention it adds.
    try:
        os.setpriority(os.PRIO_PROCESS, pid, 0)
    except PermissionError:
        subprocess.run(["renice", "-n", "0", "-p", str(pid)], capture_output=True)


def _spin_worker(stop_flag, burn_rounds: int, pin_cpu: int, duty: int = 100) -> None:
    # A Python loop yields, so 12 workers never fill one core. The C helper
    # busy-waits on the pinned CPU and exits on SIGTERM.
    proc = subprocess.Popen([os.path.join(LOAD_DIR, "spin_load"), str(pin_cpu), str(duty)])
    # _renice_normal(proc.pid)  # kept at the caller's priority for this run
    while not stop_flag.value and proc.poll() is None:
        time.sleep(0.2)
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()
    if burn_rounds < 0:
        stop_flag.value = stop_flag.value


class CpuLoadManager:
    """Overcommit spin workers on CPUs 0-19."""

    def __init__(self):
        self.ctx = multiprocessing.get_context("fork")
        self.stop_flag = None
        self.procs = []

    def start(self, n_workers: int, burn_rounds: int = 64, pin_cpu: int = 0,
              extra_workers: int = 0, extra_pin_cpu: int = 20, wake_gap: int = 0,
              duty: int = 100) -> None:
        self.stop_flag = self.ctx.Value("b", False)
        self.procs = []
        for _ in range(n_workers):
            proc = self.ctx.Process(
                target=_spin_worker, args=(self.stop_flag, burn_rounds, pin_cpu, duty)
            )
            proc.start()
            self.procs.append(proc)
        for _ in range(extra_workers):
            proc = self.ctx.Process(
                target=_wake_worker, args=(self.stop_flag, extra_pin_cpu, wake_gap)
            )
            proc.start()
            self.procs.append(proc)

    def stop(self) -> None:
        # Signal first and let the worker SIGTERM its spin_load child. Killing
        # the worker outright orphans the C process, which keeps burning CPU 0
        # and stacks on top of the next phase.
        if self.stop_flag is not None:
            self.stop_flag.value = True
        for proc in self.procs:
            try:
                proc.join(timeout=5)
            except Exception:
                pass
        for proc in self.procs:
            try:
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1)
            except Exception:
                pass
        self.procs.clear()
        if getattr(self, "gpu", None) is not None:
            self.gpu.stop()
            self.gpu = None


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

    def warmup(self) -> None:
        self.prev_stats = self.read_stats()
        self.initialized = True

    def get_usage(self) -> float:
        curr = self.read_stats()
        if not self.initialized:
            self.prev_stats = curr
            self.initialized = True
            return 0.0
        total = 0.0
        count = 0
        for idx in ALLOWED_CPU_SET:
            if idx in self.prev_stats and idx in curr:
                delta_active = curr[idx][0] - self.prev_stats[idx][0]
                delta_total = curr[idx][1] - self.prev_stats[idx][1]
                if delta_total > 0:
                    total += delta_active / delta_total * 100.0
                    count += 1
        self.prev_stats = curr
        return round(total / count, 1) if count > 0 else 0.0


class GpuLoadManager:
    """Matmul workers on GPU 0. Same shapes as the cifar10 scheduling test."""

    def __init__(self):
        self.procs = []
        self.stop_file = None
        self.script_file = None

    def start(self, n_workers: int, mem_gb: float, oom_style: bool = False) -> None:
        self.stop_file = tempfile.mktemp(suffix=".stop")
        script = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        script.write(OOM_WORKER_SCRIPT if oom_style else NEW_WORKER_SCRIPT)
        script.flush()
        script.close()
        self.script_file = script.name

        self.procs = []
        for _ in range(n_workers):
            proc = subprocess.Popen(
                [
                    "taskset", "-c", "1-19",
                    sys.executable, self.script_file,
                    str(mem_gb), self.stop_file,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.procs.append(proc)

    def stop(self) -> None:
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


def get_gpu_util(gpu_id: int = 0):
    try:
        result = subprocess.run(
            [
                "nvidia-smi", f"--id={gpu_id}",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        parts = [part.strip() for part in result.stdout.strip().split(",")]
        return float(parts[0]), float(parts[1])
    except Exception:
        return 0.0, 0.0
