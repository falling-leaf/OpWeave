#!/usr/bin/env python3
"""Regression test for HotpotQA reasoning STEP2 device selection."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from morphingdb_test.config import db_config


LOG_PATH = Path("/home/why/dbagent/pg_base/data/log") / f"postgresql-{time.strftime('%Y-%m-%d')}.log"


def run_reasoning_query(n: int = 1) -> None:
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute(
        f"SELECT unnest(db_agent_single('reasoning', sub_table.reasoning_vec)) AS score "
        f"FROM (SELECT * FROM reasoning_vector_test ORDER BY id LIMIT {n}) AS sub_table;"
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def test_hotpotqa_step2_uses_gpu():
    assert LOG_PATH.exists(), f"log file not found: {LOG_PATH}"
    before_size = LOG_PATH.stat().st_size

    rows = run_reasoning_query(1)
    time.sleep(0.5)

    with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        f.seek(before_size)
        tail = f.read()

    assert len(rows) == 1, rows
    assert "Execution task: deberta_reader, gpu" in tail, tail[-4000:]


if __name__ == "__main__":
    test_hotpotqa_step2_uses_gpu()
    print("ok")
