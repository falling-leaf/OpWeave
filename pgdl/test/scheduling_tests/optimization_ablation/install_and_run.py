#!/usr/bin/env python3
"""Install one ablation binary, restart PostgreSQL, then run the three workloads.

NOT safe to launch while another agent is using this PostgreSQL. The restart
kills every backend on the instance. This file is the runner for a later
round; ``run_ablation.py`` is the piece that can also be driven by hand
after a manual install.

Expected layout (produced by prepare_binary.py):

    optimization_ablation/bin/pgdl_opt_on.so
    optimization_ablation/bin/pgdl_opt_off.so

The live extension is resolved from pg_config --pkglibdir, falling back to
pgdl/pgdl.so in the source tree. The previous file is copied aside once,
as pgdl.so.bak.optimization_ablation, and left there. This script never
deletes that backup.

Usage:

    python install_and_run.py --optimization off --dry-run
    python install_and_run.py --optimization on
    python install_and_run.py --optimization off --workloads year_predict,cifar10
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]  # .../dbagent
PG_BIN = REPO_ROOT / "pg_base" / "bin"
PG_DATA = REPO_ROOT / "pg_base" / "data"
PG_LIBDIR = REPO_ROOT / "pg_base" / "lib" / "postgresql"
BIN_DIR = HERE / "bin"
BACKUP_NAME = "pgdl.so.bak.optimization_ablation"


def live_extension() -> Path:
    """The running server is pg_base, and $libdir/pgdl resolves here."""
    candidate = PG_LIBDIR / "pgdl.so"
    if candidate.exists():
        return candidate
    raise SystemExit(f"could not locate the installed extension at {candidate}")


def install(which: str, dry_run: bool) -> Path:
    binary = BIN_DIR / f"pgdl_opt_{which}.so"
    if not binary.exists():
        raise SystemExit(f"missing {binary}; run prepare_binary.py --which {which} first")
    target = live_extension()
    backup = target.with_name(BACKUP_NAME)
    print(f"[install] source:  {binary}")
    print(f"[install] target:  {target}")
    print(f"[install] backup:  {backup}")
    if dry_run:
        print("[install] dry-run, not copying and not restarting")
        return target
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"[install] saved previous extension to {backup}")
    else:
        print(f"[install] backup already exists, leaving it untouched: {backup}")
    shutil.copy2(binary, target)
    nested = PG_LIBDIR / "pgdl" / "pgdl.so"
    if nested.exists() and not dry_run:
        print(f"[install] not touching the extra copy at {nested}")
    return target


def restart_postgres(dry_run: bool) -> None:
    pg_ctl = PG_BIN / "pg_ctl"
    if not pg_ctl.exists():
        raise SystemExit(f"missing {pg_ctl}")
    cmd = [str(pg_ctl), "-D", str(PG_DATA), "restart", "-m", "fast"]
    print("[install] restart:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def run_workloads(which: str, workloads: list[str], extra: list[str], dry_run: bool) -> None:
    runner = HERE / "run_ablation.py"
    for workload in workloads:
        cmd = [
            sys.executable,
            str(runner),
            "--optimization", which,
            "--workload", workload,
            *extra,
        ]
        print("[run]", " ".join(cmd))
        if dry_run:
            continue
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization", choices=("on", "off"), required=True)
    parser.add_argument(
        "--workloads",
        default="year_predict,cifar10,hotpotqa",
        help="Comma-separated workload keys",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-restart", action="store_true",
                        help="Install the .so but do not restart PostgreSQL")
    args, extra = parser.parse_known_args()

    workloads = [item.strip() for item in args.workloads.split(",") if item.strip()]
    install(args.optimization, args.dry_run)
    if not args.skip_restart:
        restart_postgres(args.dry_run)
    run_workloads(args.optimization, workloads, extra, args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
