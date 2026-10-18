#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


METHOD_MAP = {
    "dbagent": "DBAgent",
    "morphingdb_baseline": "morphingdb",
    "hermes_baseline": "hermes",
    "neurdb_baseline": "neurdb",
    "gendb_baseline": "gendb",
}

SCALES = (1, 5, 10, 20)
FIELDNAMES = [
    "metric",
    "dataset",
    "method",
    "scale",
    "avg_latency_s",
    "min_latency_s",
    "max_latency_s",
]


def load_results(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def export_rows(results: dict) -> list[dict]:
    rows = []
    for raw_method in sorted(METHOD_MAP):
        if raw_method not in results:
            continue
        method = METHOD_MAP[raw_method]
        for scale in SCALES:
            metrics = results[raw_method].get(str(scale)) or results[raw_method].get(scale)
            if not metrics or metrics.get("avg") is None:
                continue
            rows.append(
                {
                    "metric": "latency",
                    "dataset": "musique",
                    "method": method,
                    "scale": scale,
                    "avg_latency_s": float(metrics["avg"]),
                    "min_latency_s": float(metrics["min"]),
                    "max_latency_s": float(metrics["max"]),
                }
            )
    rows.sort(key=lambda row: (row["method"], row["scale"]))
    return rows


def validate_monotonic(rows: list[dict]) -> list[str]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append((int(row["scale"]), float(row["avg_latency_s"])))

    failures = []
    for method, values in grouped.items():
        values.sort()
        if any(curr < prev for (_, prev), (_, curr) in zip(values, values[1:])):
            failures.append(method)
    return sorted(failures)


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strict-monotonic", action="store_true")
    args = parser.parse_args()

    results = load_results(args.input)
    rows = export_rows(results)
    failures = validate_monotonic(rows)

    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    if failures:
        print(f"Non-monotonic methods: {', '.join(failures)}")
        if args.strict_monotonic:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
