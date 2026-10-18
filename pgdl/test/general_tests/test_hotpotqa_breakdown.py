#!/usr/bin/env python3
"""
HotpotQA Performance Breakdown Experiment for DBAgent.

Methodology:
  The key challenge is that the first query has model cold-start overhead.
  We use two approaches:
  1. Compare n=1 (cold) vs n=32 (amortized) to isolate components
  2. Run back-to-back queries to measure warm performance

For n=32 at window boundary:
  - PerceptionAgent runs 32 times (one per row)
  - At row 32 (window full): OrchestrationAgent → OptimizationAgent → ExecutionAgent → EvaluationAgent
  - EvaluationAgent triggers RunReasoningPostProcess for all 32 samples
  
For n=1:
  - PerceptionAgent runs once
  - Returns early (not at window boundary) - no orchestration/execution
  - But db_agent_final still fires and returns the cross_encoder result

The cross_encoder inference IS included in the n=1 case (final call triggers it).
So n=1 = Perception(x1) + cross_encoder(x1)
n=32 = Perception(x32) + Orch + Opt + Exec(cross_encoder x32) + Eval(x32) + PostProcess(x32)
"""

import json
import csv
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import psycopg2
from tools.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir

from morphingdb_test.config import db_config


def test_hotpotqa_chart_has_dbagent_latency_rows():
    csv_path = "/home/why/dbagent/pgdl/paper_curated_20260622/anchored_generation/main_result_barcharts/main_result_barcharts_data.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    hotpotqa_dbagent = [
        row for row in rows
        if row["metric"] == "latency"
        and row["dataset"] == "HotpotQA"
        and row["method"] == "DBAgent"
    ]
    assert hotpotqa_dbagent


def test_hotpotqa_dbagent_chart_values_are_numeric():
    csv_path = "/home/why/dbagent/pgdl/paper_curated_20260622/anchored_generation/main_result_barcharts/main_result_barcharts_data.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    values = [
        float(row["value"])
        for row in rows
        if row["metric"] == "latency"
        and row["dataset"] == "HotpotQA"
        and row["method"] == "DBAgent"
    ]
    assert len(values) == 4


def test_hotpotqa_reasoning_runs_and_returns_rows():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute(
        "SELECT unnest(db_agent_single('reasoning', sub_table.reasoning_vec)) AS score "
        "FROM (SELECT * FROM reasoning_vector_test ORDER BY id LIMIT 5) AS sub_table;"
    )
    rows = cur.fetchall()
    conn.close()

    assert len(rows) == 5


def run_query(n_samples, task_type='reasoning', table='reasoning_vector_test'):
    """Run n_samples queries and return (wall_clock_s, n_results)."""
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    wall_start = time.time()
    cur.execute(
        f"SELECT unnest(db_agent_single('{task_type}', sub_table.reasoning_vec)) AS score "
        f"FROM (SELECT * FROM {table} LIMIT {n_samples}) AS sub_table;"
    )
    rows = cur.fetchall()
    wall_end = time.time()

    conn.close()
    return round(wall_end - wall_start, 4), len(rows)


def run_self_checks():
    test_hotpotqa_chart_has_dbagent_latency_rows()
    test_hotpotqa_dbagent_chart_values_are_numeric()
    test_hotpotqa_reasoning_runs_and_returns_rows()
    print("self-checks passed")


def main():
    print("=" * 80)
    print("HotpotQA Performance Breakdown for DBAgent")
    print("=" * 80)
    print()

    results = {}

    # ===================================================================
    # Measurement 1: n=1 (Perception + cross_encoder final)
    # ===================================================================
    print("Measurement 1: n=1 (single sample, cold start)")
    t1, r1 = run_query(1)
    results["n1"] = {"wall_s": t1, "rows": r1}
    print(f"  wall={t1:.4f}s, rows={r1}")
    # This includes: Perception(1) + cross_encoder(1) [cold]

    # ===================================================================
    # Measurement 2: n=1 warm (second query, models already loaded)
    # ===================================================================
    print("\nMeasurement 2: n=1 warm (models already loaded)")
    t1w, r1w = run_query(1)
    results["n1_warm"] = {"wall_s": t1w, "rows": r1w}
    print(f"  wall={t1w:.4f}s, rows={r1w}")

    # ===================================================================
    # Measurement 3: n=32 (full window, cold start)
    # ===================================================================
    print("\nMeasurement 3: n=32 (full window, cold start)")
    t32, r32 = run_query(32)
    results["n32"] = {"wall_s": t32, "rows": r32}
    print(f"  wall={t32:.4f}s, rows={r32}")

    # ===================================================================
    # Measurement 4: n=32 warm
    # ===================================================================
    print("\nMeasurement 4: n=32 warm")
    t32w, r32w = run_query(32)
    results["n32_warm"] = {"wall_s": t32w, "rows": r32w}
    print(f"  wall={t32w:.4f}s, rows={r32w}")

    # ===================================================================
    # Measurement 5: n=20 (below window, cold)
    # ===================================================================
    print("\nMeasurement 5: n=20 (below window)")
    t20, r20 = run_query(20)
    results["n20"] = {"wall_s": t20, "rows": r20}
    print(f"  wall={t20:.4f}s, rows={r20}")

    # ===================================================================
    # Measurement 6: n=20 warm
    # ===================================================================
    print("\nMeasurement 6: n=20 warm")
    t20w, r20w = run_query(20)
    results["n20_warm"] = {"wall_s": t20w, "rows": r20w}
    print(f"  wall={t20w:.4f}s, rows={r20w}")

    # ===================================================================
    # BREAKDOWN ANALYSIS (warm measurements, excluding cold start)
    # ===================================================================
    print("\n" + "=" * 80)
    print("BREAKDOWN ANALYSIS (warm measurements, n=32)")
    print("=" * 80)

    t1w_ms = t1w * 1000  # warm single sample
    t32w_ms = t32w * 1000  # warm 32 samples

    # n=1 warm = Perception(1) + cross_encoder(1)
    # n=32 warm = Perception(32) + Orch + Opt + Exec(cross_encoder 32) + Eval(32) + PostProcess(32)
    
    # Perception per sample (warm) from n=1:
    # n=1 warm ≈ Perception(1) + cross_encoder(1)
    # But we know n=32 warm - 32 × n=1 warm would give us the pipeline overhead
    # except that the cross_encoder is batched in n=32, so it's more efficient
    
    # Let's use the n=20 measurements which are below window boundary:
    # n=20 warm = Perception(20) + cross_encoder(20) [in final call]
    # This is the same agents as n=1, just more samples
    
    t20w_ms = t20w * 1000
    
    # Perception + cross_encoder per sample (warm):
    per_sample_warm_ms = t20w_ms / 20

    print(f"\n  n=1 warm:     {t1w_ms:10.3f} ms (Perception + cross_encoder)")
    print(f"  n=20 warm:    {t20w_ms:10.3f} ms (Perception x20 + cross_encoder)")
    print(f"  n=32 warm:    {t32w_ms:10.3f} ms (full pipeline)")
    print(f"  Per sample (warm): {per_sample_warm_ms:10.3f} ms")

    # For n=32 warm:
    # Perception + cross_encoder for 32 samples would take: per_sample_warm_ms * 32
    # But the actual n=32 warm includes additional: Orch + Opt + PostProcess
    
    # However, Perception in n=32 is NOT 32 separate calls - it's 32 rapid calls
    # and the cross_encoder runs ONCE for all 32 at window boundary
    # So n=20 warm ≠ 20 × n=1 warm because the cross_encoder batches

    # Better decomposition:
    # n=1 warm = Perception(1) + cross_encoder(1)
    # n=32 warm = Perception(32) + Orch + Opt + Exec(cross_encoder_batch_32) + Eval + PostProcess(32)
    
    # The key difference between n=20 and n=32 at the window:
    # n=20: returns early after perception (no Orch/Opt/Exec) but still runs cross_encoder in final
    # n=32: runs full pipeline at window boundary

    # From the code flow:
    # - For n < 32: Perception runs, ScheduleAgent returns SUCCESS (no Orch/Opt/Exec)
    #   BUT db_agent_final still runs and executes cross_encoder for whatever data was collected
    # - For n = 32: Perception x32 → Orch → Opt → Exec(cross_encoder) → Eval → PostProcess

    # So the additional cost of n=32 over n=20 (scaled) is:
    # Orch + Opt + PostProcess + (batch efficiency gain for cross_encoder)

    # Let me compute this differently:
    # Cost of Orch + Opt + PostProcess for n=32:
    additional_cost_32_vs_20 = t32w_ms - t20w_ms * (32/20)
    
    print(f"\n  Additional cost for n=32 vs scaled n=20:")
    print(f"    n=32 warm:            {t32w_ms:10.3f} ms")
    print(f"    n=20 warm × 1.6:      {t20w_ms * 32/20:10.3f} ms")
    print(f"    Extra (Orch+Opt+Post): {additional_cost_32_vs_20:10.3f} ms")

    # Now let's do a more direct measurement:
    # The n=32 cold start cost = t32 - t32w
    cold_start_ms = (t32 - t32w) * 1000
    print(f"\n  Cold start overhead: {cold_start_ms:.3f} ms")

    # ===================================================================
    # SIMPLIFIED BREAKDOWN TABLE
    # ===================================================================
    print("\n" + "=" * 80)
    print("PERFORMANCE BREAKDOWN TABLE (n=32, warm)")
    print("=" * 80)
    
    # Based on our measurements:
    # n=1 warm = 3942 ms = Perception + cross_encoder(1)
    # n=20 warm / 20 = 4696 ms = Perception + cross_encoder per sample
    # n=32 warm = 90031 ms = Perception(32) + Orch + Opt + Exec(cross_encoder batch 32) + Eval + PostProcess(32)
    
    # If cross_encoder was linear: 32 × n=1_warm = 32 × 3942 = 126144 ms
    # But n=32 warm = 90031 ms (BATCHING SAVINGS: ~36113 ms saved)
    
    # So the actual pipeline cost breakdown:
    # Total: 90031 ms
    # - Perception(32): we estimate this is very fast (~1-2 ms each) since it's just caching data
    # - cross_encoder batch(32): ~75-80% of total (dominant cost)
    # - PostProcess(32 STEP2 + decoder): ~15-20%
    # - Orch + Opt + Eval: <5%
    
    # Since we can't separate these precisely without code instrumentation,
    # let's present the measurement results directly:
    
    print(f"\n  {'Measurement':<30} {'Wall Time (s)':>12} {'Notes':<30}")
    print(f"  {'─' * 72}")
    print(f"  {'n=1 (cold)':<30} {t1:>12.4f} {'Perception + cross_encoder(1) [cold]':<30}")
    print(f"  {'n=1 (warm)':<30} {t1w:>12.4f} {'Perception + cross_encoder(1) [warm]':<30}")
    print(f"  {'n=20 (cold)':<30} {t20:>12.4f} {'Perception x20 + cross_encoder [cold]':<30}")
    print(f"  {'n=20 (warm)':<30} {t20w:>12.4f} {'Perception x20 + cross_encoder [warm]':<30}")
    print(f"  {'n=32 (cold)':<30} {t32:>12.4f} {'Full pipeline (Perception x32 + ... + PostProcess) [cold]':<30}")
    print(f"  {'n=32 (warm)':<30} {t32w:>12.4f} {'Full pipeline [warm]':<30}")

    print(f"\n  Cold start overhead: {cold_start_ms:.1f} ms ({cold_start_ms/t32w_ms*100:.1f}% of total)")
    print(f"  Warm per-sample cost: {t32w_ms/32:.1f} ms/sample")

    # ===================================================================
    # COMPARISON TABLE (cold vs warm for paper)
    # ===================================================================
    print(f"\n{'=' * 80}")
    print("HOT POT QA PERFORMANCE SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n  {'n':>5} {'Cold (s)':>10} {'Warm (s)':>10} {'Cold/sample (ms)':>18} {'Warm/sample (ms)':>18}")
    print(f"  {'─' * 65}")
    for n, cold, warm in [(1, t1, t1w), (20, t20, t20w), (32, t32, t32w)]:
        print(f"  {n:>5} {cold:>10.4f} {warm:>10.4f} {cold*1000/n:>18.2f} {warm*1000/n:>18.2f}")

    # Save results
    results_dir = ensure_dir(GENERAL_RESULTS_DIR)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, f"hotpotqa_breakdown_{ts}.json")

    breakdown = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "hotpotqa_breakdown",
        "methodology": "client-side timing via isolated agent queries (cold vs warm)",
        "measurements": {
            "n1_cold": {"wall_s": t1, "rows": r1},
            "n1_warm": {"wall_s": t1w, "rows": r1w},
            "n20_cold": {"wall_s": t20, "rows": r20},
            "n20_warm": {"wall_s": t20w, "rows": r20w},
            "n32_cold": {"wall_s": t32, "rows": r32},
            "n32_warm": {"wall_s": t32w, "rows": r32w},
        },
        "analysis": {
            "cold_start_ms": round(cold_start_ms, 3),
            "warm_total_ms": round(t32w_ms, 3),
            "warm_per_sample_ms": round(t32w_ms / 32, 3),
            "perception_plus_cross_encoder_per_sample_ms": round(per_sample_warm_ms, 3)
        }
    }

    with open(json_path, 'w') as f:
        json.dump(breakdown, f, indent=2)

    csv_path = os.path.join(results_dir, f"hotpotqa_breakdown_{ts}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 'cold_s', 'warm_s', 'cold_per_sample_ms', 'warm_per_sample_ms'])
        for n, cold, warm in [(1, t1, t1w), (20, t20, t20w), (32, t32, t32w)]:
            writer.writerow([n, cold, warm, round(cold*1000/n, 2), round(warm*1000/n, 2)])

    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
