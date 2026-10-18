#!/usr/bin/env python3
import json
import os
import sys
import time
from datetime import datetime

import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from morphingdb_test.config import db_config
from tools.test_result_paths import GENERAL_RESULTS_DIR, ensure_dir


ROW_COUNTS = [2000, 4000, 6000, 8000, 10000]

DATASETS = {
    "cifar-10": {
        "original": "select predict_batch_float8('googlenet_cifar10', 'gpu', image_vector) over (rows between current row and 31 following) from cifar_image_vector_table limit {row_count};",
        "dbagent": "select unnest(db_agent_single('image_classification', sub_table.image_vector)) AS score FROM (SELECT * FROM cifar_image_vector_table limit {row_count}) AS sub_table;",
    },
    "stanford_dogs": {
        "original": "select predict_batch_float8('alexnet_stanford_dogs', 'gpu', image_vector) over (rows between current row and 31 following) from stanford_dogs_image_vector_table limit {row_count};",
        "dbagent": "select unnest(db_agent_single('image_classification', sub_table.image_vector)) AS score FROM (SELECT * FROM stanford_dogs_image_vector_table limit {row_count}) AS sub_table;",
    },
}


def run_once(sql):
    conn = psycopg2.connect(**db_config)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("select register_process();")
    cur.execute("SET statement_timeout = '240s';")
    start = time.time()
    cur.execute(sql)
    rows = cur.fetchall()
    latency = time.time() - start
    conn.close()
    return latency, len(rows)


def main():
    results = {
        "timestamp": datetime.now().isoformat(),
        "row_counts": ROW_COUNTS,
        "datasets": {},
    }

    for dataset, queries in DATASETS.items():
        print(f"=== {dataset} ===")
        ds_results = {}
        for kind, template in queries.items():
            kind_results = []
            print(kind)
            for row_count in ROW_COUNTS:
                sql = template.format(row_count=row_count)
                latency, rows = run_once(sql)
                entry = {"row_count": row_count, "latency": latency, "rows": rows}
                kind_results.append(entry)
                print(f"  {row_count}: {latency:.4f}s rows={rows}")
            ds_results[kind] = kind_results
        results["datasets"][dataset] = ds_results

    out_dir = ensure_dir(GENERAL_RESULTS_DIR)
    out_path = os.path.join(out_dir, "image_single_test_probe.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
