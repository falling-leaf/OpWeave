from export_musique_baseline_csv import export_rows, validate_monotonic


def test_export_rows_maps_requested_methods_only():
    results = {
        "dbagent": {
            "1": {"avg": 1.0, "min": 0.9, "max": 1.1},
            "5": {"avg": 5.0, "min": 4.9, "max": 5.1},
        },
        "morphingdb_baseline": {
            "1": {"avg": 1.5, "min": 1.4, "max": 1.6},
            "5": {"avg": 5.5, "min": 5.4, "max": 5.6},
        },
        "hermes_baseline": {
            "1": {"avg": 2.0, "min": 1.9, "max": 2.1},
            "5": {"avg": 6.0, "min": 5.9, "max": 6.1},
        },
        "neurdb_baseline": {
            "1": {"avg": 3.0, "min": 2.9, "max": 3.1},
            "5": {"avg": 7.0, "min": 6.9, "max": 7.1},
        },
        "gendb_baseline": {
            "1": {"avg": 4.0, "min": 3.9, "max": 4.1},
            "5": {"avg": 8.0, "min": 7.9, "max": 8.1},
        },
    }

    rows = export_rows(results)

    assert [row["method"] for row in rows[:4]] == ["DBAgent", "DBAgent", "gendb", "gendb"]
    assert {row["method"] for row in rows} == {"DBAgent", "gendb", "hermes", "morphingdb", "neurdb"}
    assert all(row["dataset"] == "musique" for row in rows)
    assert all(row["metric"] == "latency" for row in rows)


def test_validate_monotonic_reports_non_increasing_method():
    rows = [
        {"method": "DBAgent", "scale": 1, "avg_latency_s": 1.0},
        {"method": "DBAgent", "scale": 5, "avg_latency_s": 4.0},
        {"method": "DBAgent", "scale": 10, "avg_latency_s": 3.0},
    ]

    failures = validate_monotonic(rows)

    assert failures == ["DBAgent"]
