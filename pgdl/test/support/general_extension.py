#!/usr/bin/env python3
"""Install the optimization-off DBAgent binary for non-scheduling tests.

ADD_OPTIMIZATION_LOGIC is compiled into pgdl.so. Scheduling tests install the
on binary themselves and must keep doing that. Every other measurement wants
the off binary: ScheduleAgent then skips OptimizationAgent, and
OrchestrationAgent keeps the device it picked for the task type (series on
CPU, image and reasoning on GPU).

Importing this module does nothing. Call ensure_opt_off() at the start of a
general test, before opening a connection.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TEST_ROOT.parents[1]
PG_BIN = REPO_ROOT / "pg_base" / "bin"
PG_DATA = REPO_ROOT / "pg_base" / "data"
PG_LIB = REPO_ROOT / "pg_base" / "lib" / "postgresql" / "pgdl.so"
OPT_OFF = (
    TEST_ROOT
    / "scheduling_tests"
    / "optimization_ablation"
    / "bin"
    / "pgdl_opt_off.so"
)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_opt_off() -> None:
    """Install pgdl_opt_off.so if the live extension is a different binary.

    Restarts PostgreSQL only when the file actually changes. A restart drops
    every backend on this instance, so do not call this while a scheduling
    run is in progress.
    """
    if not OPT_OFF.exists():
        raise SystemExit(
            f"missing {OPT_OFF}; build it with "
            "scheduling_tests/optimization_ablation/prepare_binary.py --which off"
        )
    if not PG_LIB.exists():
        raise SystemExit(f"missing installed extension {PG_LIB}")
    if _md5(PG_LIB) == _md5(OPT_OFF):
        print(f"[general] pgdl.so already matches {OPT_OFF.name}")
        return
    print(f"[general] installing {OPT_OFF.name} over {PG_LIB}")
    PG_LIB.write_bytes(OPT_OFF.read_bytes())
    pg_ctl = PG_BIN / "pg_ctl"
    subprocess.run(
        [str(pg_ctl), "-D", str(PG_DATA), "restart", "-m", "fast"],
        check=True,
    )
    print("[general] restarted PostgreSQL with ADD_OPTIMIZATION_LOGIC off")
