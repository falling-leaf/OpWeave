#!/usr/bin/env python3
"""Turn ablation JSON into the extra Table 3 row. Does not edit the paper.

Reads pgdl/test/support/results/scheduling/optimization_ablation/*.json
written by run_ablation.py. Emits:

  optimization_ablation/table3_opt_off_row.tex
  optimization_ablation/table3_opt_off_row.json

The existing DBAgent row in tables/scheduling_load.tex is the optimization-on
condition (ADD_OPTIMIZATION_LOGIC was enabled for those scheduling runs).
This row is the off condition: the device stays at the orchestration default
(year_predict on CPU, cifar10 on GPU, HotpotQA on GPU).

Latency is the minimum successful timed run, converted to seconds, with the
same rounding the current table uses: 3 decimals for year_predict, 1 decimal
for cifar10 and HotpotQA.

Usage:

    python fill_table_row.py
    python fill_table_row.py --results-dir /path/to/jsons
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = (
    HERE.parents[1] / "support" / "results" / "scheduling" / "optimization_ablation"
)

COLUMNS = ("year_predict", "cifar10", "hotpotqa")
PHASES = ("low", "medium", "high")
DECIMALS = {"year_predict": 3, "cifar10": 1, "hotpotqa": 1}


def _latest_by_key(results_dir: Path) -> dict[tuple[str, str], Path]:
    found: dict[tuple[str, str], Path] = {}
    if not results_dir.is_dir():
        return found
    for path in sorted(results_dir.glob("*_opt_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        workload = payload.get("workload")
        optimization = payload.get("optimization")
        if workload in COLUMNS and optimization in ("on", "off"):
            # sorted() is chronological because the stamp is YYYYMMDD_HHMMSS,
            # so the last write for a key wins.
            found[(workload, optimization)] = path
    return found


def _cell(payload: dict, phase: str, decimals: int) -> str:
    for item in payload.get("phases", []):
        if item.get("phase") != phase:
            continue
        seconds = item.get("min_latency_s")
        if seconds is None and item.get("min_latency_ms") is not None:
            seconds = item["min_latency_ms"] / 1000.0
        if seconds is None:
            return "---"
        return f"{seconds:.{decimals}f}"
    return "---"


def _row(label: str, cells: list[str]) -> str:
    body = " & ".join(cells)
    return (
        f"  {label}\n"
        f"    & {body} \\\\"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    latest = _latest_by_key(args.results_dir)
    off_cells = []
    on_cells = []
    used = {}
    for workload in COLUMNS:
        decimals = DECIMALS[workload]
        off_path = latest.get((workload, "off"))
        on_path = latest.get((workload, "on"))
        used[workload] = {
            "off": str(off_path) if off_path else None,
            "on": str(on_path) if on_path else None,
        }
        if off_path is None:
            off_cells.extend(["---", "---", "---"])
        else:
            payload = json.loads(off_path.read_text(encoding="utf-8"))
            off_cells.extend(_cell(payload, phase, decimals) for phase in PHASES)
        if on_path is None:
            on_cells.extend(["---", "---", "---"])
        else:
            payload = json.loads(on_path.read_text(encoding="utf-8"))
            on_cells.extend(_cell(payload, phase, decimals) for phase in PHASES)

    # Group as three triples so the row lines up with the table's cmidrules.
    def grouped(cells: list[str]) -> str:
        triples = [" & ".join(cells[i:i + 3]) for i in range(0, 9, 3)]
        return "\n    & " + "\n    & ".join(triples) + " \\\\"

    tex = "\n".join([
        "% Extra row for tables/scheduling_load.tex (Table 3).",
        "% optimization off: device fixed at the orchestration default",
        "%   year_predict -> cpu, cifar10 -> gpu, HotpotQA -> gpu.",
        "% Paste under the \\\\modelname{} row. Do not replace that row;",
        "% the published numbers are the optimization-on condition.",
        "  \\modelname{} (w/o opt.)" + grouped(off_cells),
        "% Optional check row from a fresh optimization-on rerun.",
        "% The published \\\\modelname{} row stays the paper number;",
        "% this one only confirms the rerun.",
        "%   \\modelname{} (opt. on, rerun)" + grouped(on_cells),
        "",
    ])

    if not any(path for path in used.values() for path in path.values() if path):
        print(tex)
        print(f"no result JSON under {args.results_dir}; not writing a row file")
        return

    tex_path = HERE / "table3_opt_off_row.tex"
    json_path = HERE / "table3_opt_off_row.json"
    tex_path.write_text(tex, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "results_dir": str(args.results_dir),
                "sources": used,
                "off_cells_low_med_high_by_workload": {
                    workload: off_cells[i:i + 3]
                    for i, workload in zip(range(0, 9, 3), COLUMNS)
                },
                "on_rerun_cells_low_med_high_by_workload": {
                    workload: on_cells[i:i + 3]
                    for i, workload in zip(range(0, 9, 3), COLUMNS)
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(tex)
    print(f"wrote {tex_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
