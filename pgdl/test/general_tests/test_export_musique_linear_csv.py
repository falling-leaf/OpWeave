from export_musique_baseline_csv import export_rows, validate_monotonic


def test_export_rows_supports_linear_benchmark_method_names():
    results = {
        "dbagent": {"1": {"avg": 1.0, "min": 0.9, "max": 1.1}},
        "morphingdb_baseline": {"1": {"avg": 2.0, "min": 1.9, "max": 2.1}},
        "hermes_baseline": {"1": {"avg": 3.0, "min": 2.9, "max": 3.1}},
        "neurdb_baseline": {"1": {"avg": 4.0, "min": 3.9, "max": 4.1}},
        "gendb_baseline": {"1": {"avg": 5.0, "min": 4.9, "max": 5.1}},
    }

    rows = export_rows(results)

    assert [row["method"] for row in rows] == ["DBAgent", "gendb", "hermes", "morphingdb", "neurdb"]


def test_validate_monotonic_accepts_increasing_linear_rows():
    rows = [
        {"method": "morphingdb", "scale": 1, "avg_latency_s": 1.0},
        {"method": "morphingdb", "scale": 5, "avg_latency_s": 5.0},
        {"method": "morphingdb", "scale": 10, "avg_latency_s": 10.0},
        {"method": "morphingdb", "scale": 20, "avg_latency_s": 20.0},
    ]

    assert validate_monotonic(rows) == []
