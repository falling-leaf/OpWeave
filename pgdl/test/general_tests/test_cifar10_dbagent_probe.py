#!/usr/bin/env python3
import sys
import time

import psycopg2

sys.path.insert(0, "/home/why/dbagent/pgdl/test")
from morphingdb_test.config import db_config


def run_query(row_count):
    sql = (
        "select unnest(db_agent_single('image_classification', sub_table.image_vector)) AS score "
        "FROM (SELECT * FROM cifar_image_vector_table limit %s) AS sub_table;"
    )
    conn = psycopg2.connect(**db_config)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("select register_process();")
    cur.execute("SET statement_timeout = '240s';")
    start = time.time()
    cur.execute(sql, (row_count,))
    rows = cur.fetchall()
    latency = time.time() - start
    conn.close()
    return latency, len(rows)


def main():
    for row_count in (2000, 4000):
        latency, rows = run_query(row_count)
        assert rows == row_count, (row_count, rows)
        print(row_count, latency, rows)


if __name__ == "__main__":
    main()
