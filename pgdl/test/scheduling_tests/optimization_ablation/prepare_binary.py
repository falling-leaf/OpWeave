#!/usr/bin/env python3
"""Build two DBAgent binaries that differ only in ADD_OPTIMIZATION_LOGIC.

The macro is a ``#define`` in ``model_agent.cpp``. A compiler ``-D`` does not
override it, because the ``#define`` is processed later and wins. This script
therefore rewrites that one line for the duration of each build and restores
the file afterwards, including on failure.

The edit is guarded: the build runs only while this process holds an exclusive
lock, and the file is restored before the lock is released. ``pgdl/build`` is
not used. Outputs go to ``optimization_ablation/build/{on,off}`` and
``optimization_ablation/bin/pgdl_opt_{on,off}.so``.

Nothing is installed. ``ALWAYS_TRAINING`` must stay false.

Usage:

    python prepare_binary.py --which both
    python prepare_binary.py --which off --jobs 8
"""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PGDL_ROOT = HERE.parents[2]  # .../pgdl
SRC_AGENT = PGDL_ROOT / "src" / "pgdl" / "model_agent.cpp"
BUILD_ROOT = HERE / "build"
BIN_DIR = HERE / "bin"
LOCK_PATH = HERE / ".prepare.lock"

PREFIX_PATH = ";".join([
    str(PGDL_ROOT.parent / "pg_base"),
    str(PGDL_ROOT / "third_party" / "libtorch"),
    str(PGDL_ROOT / "third_party" / "onnxruntime-linux-x64-1.18.1"),
])
FLAG_LINE = {
    True: "#define ADD_OPTIMIZATION_LOGIC true",
    False: "#define ADD_OPTIMIZATION_LOGIC false",
}
TRAINING_LINE = "#define ALWAYS_TRAINING false"


def _read() -> str:
    return SRC_AGENT.read_text(encoding="utf-8")


def _require_pristine(text: str) -> None:
    present = [line for line in FLAG_LINE.values() if line in text]
    if len(present) != 1:
        raise SystemExit(
            f"{SRC_AGENT} does not contain exactly one ADD_OPTIMIZATION_LOGIC "
            "define. Refusing to edit it."
        )
    if TRAINING_LINE not in text:
        raise SystemExit(
            f"{SRC_AGENT} does not keep ALWAYS_TRAINING false. "
            "Measurement builds must not train."
        )


def _swap_flag(enabled: bool) -> str:
    """Return the original text. Caller must restore it."""
    text = _read()
    _require_pristine(text)
    target = FLAG_LINE[enabled]
    if target in text:
        return text
    other = FLAG_LINE[not enabled]
    updated = text.replace(other, target, 1)
    if updated == text or target not in updated:
        raise SystemExit("failed to rewrite ADD_OPTIMIZATION_LOGIC")
    SRC_AGENT.write_text(updated, encoding="utf-8")
    return text


def _restore(original: str) -> None:
    current = _read()
    if current != original:
        SRC_AGENT.write_text(original, encoding="utf-8")


def _configure(build_dir: Path) -> None:
    if (build_dir / "CMakeCache.txt").exists():
        return
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cmake", "-S", str(PGDL_ROOT), "-B", str(build_dir),
            f"-DCMAKE_PREFIX_PATH={PREFIX_PATH}",
            "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
        ],
        check=True,
    )


def _build(build_dir: Path, jobs: int) -> Path:
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "pgdl", "-j", str(jobs)],
        check=True,
    )
    produced = build_dir / "pgdl.so"
    if not produced.exists():
        raise SystemExit(f"build finished but {produced} is missing")
    return produced


def build_one(which: str, jobs: int, original: str) -> Path:
    enabled = which == "on"
    build_dir = BUILD_ROOT / which
    print(f"[prepare] {FLAG_LINE[enabled]} -> {build_dir}")
    _configure(build_dir)
    _swap_flag(enabled)
    try:
        produced = _build(build_dir, jobs)
    finally:
        _restore(original)
    if _read() != original:
        raise SystemExit(f"failed to restore {SRC_AGENT}")
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    dest = BIN_DIR / f"pgdl_opt_{which}.so"
    shutil.copy2(produced, dest)
    print(f"[prepare] copied {produced} -> {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--which", choices=("on", "off", "both"), default="both")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()

    original = _read()
    _require_pristine(original)
    targets = ("off", "on") if args.which == "both" else (args.which,)

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            built = [build_one(which, args.jobs, original) for which in targets]
        finally:
            _restore(original)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    if _read() != original:
        raise SystemExit(f"{SRC_AGENT} was not restored")
    print("[prepare] source restored. Binaries are NOT installed.")
    for path in built:
        print(f"  {path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
