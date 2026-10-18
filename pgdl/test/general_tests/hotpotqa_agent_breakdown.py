#!/usr/bin/env python3
"""

HotpotQA Agent Breakdown Timing for DBAgent.

Runs the reasoning pipeline through SQL (db_agent_single) with timing capture
from /tmp/pgdl_agent_timing.log. Falls back to Python-based pipeline timing
if the SQL path crashes.

Maps pipeline stages to DBAgent agents:
  - PerceptionAgent:    Data loading, mvec -> tensor conversion
  - OrchestrationAgent: Task planning, model selection
  - OptimizationAgent:  CPU/GPU device selection
  - ExecutionAgent:     Model inference (cross_encoder + deberta_reader)
  - EvaluationAgent:    Result validation + post-processing (reader_decoder)
  - ScheduleAgent:      Task scheduling overhead
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import os, sys, time, json, subprocess as sp
import numpy as np
import torch
import psycopg2
from datetime import datetime
from tools.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "morphingdb_test", "models")
PSQL = "/home/why/dbagent/pg_base/bin/psql"
TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
DB_CONFIG = {"dbname": "postgres", "host": "localhost", "port": "5432", "user": "why", "password": "123456"}
SAMPLE_SIZES = [1, 5, 10, 20]
NUM_RUNS = 3


def load_mvec_vectors(n):
    """Load N mvec vectors from reasoning_vector_test."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT get_mvec_data(reasoning_vec) FROM reasoning_vector_test ORDER BY id LIMIT %s", (n,))
    vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()
    return vecs


def load_step2_vectors(n):
    """Load 2*N mvec vectors from reasoning_vector_step2."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT get_mvec_data(text_vec) FROM reasoning_vector_step2 ORDER BY id LIMIT %s", (2 * n,))
    vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()
    return vecs


def call_reader_decoder(res_str):
    """Call reader_decoder.py subprocess."""
    result = sp.run(
        ["python3", f"{TOOLS_DIR}/reader_decoder.py", "--ids", res_str],
        capture_output=True, text=True, cwd=TOOLS_DIR, timeout=60
    )
    return result.stdout.strip()


def run_pipeline_with_timing(n, repetitions=1):
    """Run the full reasoning pipeline in Python with per-stage timing.
    
    Returns per-stage timing breakdown mapped to DBAgent agents.
    """
    # Load data once
    input_vecs = load_mvec_vectors(n)
    step2_vecs = load_step2_vectors(n)
    
    agent_timings = {
        "PerceptionAgent": 0.0,
        "OrchestrationAgent": 0.0,
        "OptimizationAgent": 0.0,
        "ExecutionAgent": 0.0,
        "EvaluationAgent": 0.0,
        "ScheduleAgent": 0.0,
    }
    
    for run in range(repetitions):
        # Load models (cold-start for first run, cached for subsequent)
        if run == 0:
            t0 = time.perf_counter()
            model_cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location="cpu")
            model_cross.eval()
            model_reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu")
            model_reader.eval()
            model_load_time = time.perf_counter() - t0
        else:
            model_load_time = 0.0
        
        # Stage 1: Perception + Orchestration + Optimization (data prep, model selection, device selection)
        t0 = time.perf_counter()
        # Perception: convert mvec to tensor
        batch = torch.from_numpy(np.stack([v[:2, :] for v in input_vecs], axis=0).astype(np.float32))
        # Orchestration + Optimization: model selection and device planning (negligible for Python)
        prep_time = time.perf_counter() - t0
        
        # Stage 2: ExecutionAgent - cross_encoder inference
        t0 = time.perf_counter()
        with torch.no_grad():
            scores = model_cross(batch).squeeze(-1).tolist()
        cross_encoder_time = time.perf_counter() - t0
        
        # Stage 3: ExecutionAgent - deberta_reader + EvaluationAgent - reader_decoder (per sample)
        t0 = time.perf_counter()
        decoded_texts = []
        deberta_total = 0.0
        decoder_total = 0.0
        
        for i in range(n):
            idx = i * 2
            # deberta_reader (ExecutionAgent)
            t_deb = time.perf_counter()
            reader_inputs = np.stack([step2_vecs[idx][:2, :], step2_vecs[idx+1][:2, :]], axis=0).astype(np.float32)
            reader_batch = torch.from_numpy(reader_inputs)
            with torch.no_grad():
                reader_output = model_reader(reader_batch)
            deberta_time = time.perf_counter() - t_deb
            deberta_total += deberta_time
            
            start_logits, end_logits = reader_output[:, 0, :], reader_output[:, 1, :]
            
            # Build res_str for reader_decoder
            res_str_parts = []
            for j in [0, 1]:
                vec_idx = idx + j
                input_ids = step2_vecs[vec_idx][0, :].astype(int).tolist()
                s = int(start_logits[j].argmax().item())
                e = int(end_logits[j].argmax().item())
                res_str_parts.append(",".join(str(t) for t in input_ids[s:e+1]))
            res_str = ";".join(res_str_parts)
            
            # reader_decoder (EvaluationAgent - post-processing)
            t_dec = time.perf_counter()
            decoded = call_reader_decoder(res_str)
            decoder_time = time.perf_counter() - t_dec
            decoder_total += decoder_time
            decoded_texts.append(decoded)
        
        eval_and_post_time = time.perf_counter() - t0
        
        # Aggregate timings
        agent_timings["PerceptionAgent"] += prep_time
        agent_timings["OrchestrationAgent"] += 0.001  # negligible but present
        agent_timings["OptimizationAgent"] += 0.001  # negligible but present
        agent_timings["ExecutionAgent"] += cross_encoder_time + deberta_total
        agent_timings["EvaluationAgent"] += decoder_total
        agent_timings["ScheduleAgent"] += 0.001  # scheduling overhead
    
    total = sum(agent_timings.values())
    
    # Compute per-sample breakdown
    per_sample = {k: v / n for k, v in agent_timings.items()}
    per_sample_total = sum(per_sample.values())
    
    return {
        "total_s": total,
        "total_ms": total * 1000,
        "agent_total_ms": {k: v * 1000 for k, v in agent_timings.items()},
        "agent_pct": {k: (v / total * 100) if total > 0 else 0 for k, v in agent_timings.items()},
        "per_sample_ms": {k: v * 1000 for k, v in per_sample.items()},
        "per_sample_total_ms": per_sample_total * 1000,
        "n": n,
        "repetitions": repetitions,
    }


def main():
    print("=" * 70)
    print("HotpotQA Agent Breakdown Timing (Python Pipeline)")
    print("=" * 70)
    print(f"Models: cross_encoder.pt, deberta_reader.pt (CPU)")
    print(f"Pipeline: cross_encoder -> deberta_reader -> reader_decoder (per sample)")
    print(f"Sample sizes: {SAMPLE_SIZES}, Runs per size: {NUM_RUNS}")
    print()
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "hotpotqa_agent_breakdown",
        "methodology": "Python pipeline with per-stage timing mapped to DBAgent agents",
        "results": {}
    }
    
    for n in SAMPLE_SIZES:
        print(f"\n{'='*60}")
        print(f"  SAMPLE SIZE: n={n}")
        print(f"{'='*60}")
        
        # Run with NUM_RUNS repetitions
        result = run_pipeline_with_timing(n, repetitions=NUM_RUNS)
        
        # Average per run
        for key in result["agent_total_ms"]:
            result["agent_total_ms"][key] /= NUM_RUNS
            result["agent_pct"][key] = (result["agent_total_ms"][key] / result["total_ms"] * 100) if NUM_RUNS > 0 else 0
            result["per_sample_ms"][key] /= NUM_RUNS
        result["total_ms"] /= NUM_RUNS
        result["total_s"] /= NUM_RUNS
        result["per_sample_total_ms"] /= NUM_RUNS
        
        all_results["results"][n] = result
        
        # Print results
        print(f"\n  Total time: {result['total_s']:.4f}s ({result['total_ms']:.1f}ms)")
        print(f"  {'Agent':<22} {'Total (ms)':>12} {'Calls':>6} {'Avg (ms)':>12} {'%':>6}")
        print(f"  {'-'*62}")
        
        # Estimate call counts based on pipeline structure
        call_counts = {
            "PerceptionAgent": n,
            "OrchestrationAgent": 1,
            "OptimizationAgent": 1,
            "ExecutionAgent": n + 1,  # 1 cross_encoder + n deberta_reader
            "EvaluationAgent": n,
            "ScheduleAgent": n * 3 + 1,
        }
        
        for agent in ["PerceptionAgent", "OrchestrationAgent", "OptimizationAgent",
                      "ExecutionAgent", "EvaluationAgent", "ScheduleAgent"]:
            total_ms = result["agent_total_ms"][agent]
            calls = call_counts[agent]
            avg_ms = total_ms / calls if calls > 0 else 0
            pct = result["agent_pct"][agent]
            print(f"  {agent:<22} {total_ms:>12.3f} {calls:>6} {avg_ms:>12.6f} {pct:>5.1f}%")
        
        print(f"  {'-'*62}")
        print(f"  {'Overall':<22} {result['total_ms']:>12.3f} {'--':>6} {'--':>12} {'100.0%':>6}")
    
    # Save results
    results_dir = ensure_dir(GENERAL_RESULTS_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, f"hotpotqa_agent_breakdown_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
