#!/usr/bin/env python3
"""Shared result directory helpers for the PGDL test workspace."""

from __future__ import annotations

from pathlib import Path

SUPPORT_ROOT = Path(__file__).resolve().parent
TEST_ROOT = SUPPORT_ROOT.parent
RESULTS_ROOT = TEST_ROOT / "results"
GENERAL_RESULTS_DIR = RESULTS_ROOT / "general"
THROUGHPUT_RESULTS_DIR = RESULTS_ROOT / "throughput"
SCHEDULING_RESULTS_DIR = RESULTS_ROOT / "scheduling"
INIT_RESULTS_DIR = RESULTS_ROOT / "init"


def ensure_dir(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
