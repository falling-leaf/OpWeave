#!/usr/bin/env python3
import re
import unittest
from unittest import mock

import test_single


class FakeCursor:
    def __init__(self):
        self._last_sql = ""

    def execute(self, sql):
        self._last_sql = sql

    def fetchall(self):
        match = re.search(r"limit\s+(\d+)", self._last_sql, re.IGNORECASE)
        if not match:
            raise AssertionError(f"could not parse row count from SQL: {self._last_sql}")
        row_count = int(match.group(1))
        return [(0,)] * row_count


class FakeConnection:
    def __init__(self):
        self.cursor_obj = FakeCursor()

    def cursor(self):
        return self.cursor_obj

    def close(self):
        return None


class MusiqueConnectionReuseTest(unittest.TestCase):
    def test_musique_ladder_reuses_one_connection(self):
        connect_calls = []

        def fake_connect(**_kwargs):
            connect_calls.append(object())
            return FakeConnection()

        musique_only = [
            query for query in test_single.NEW_SQL_QUERIES if query["name"] == "musique_db_agent"
        ]

        with mock.patch.object(test_single, "NEW_SQL_QUERIES", musique_only):
            with mock.patch.object(test_single.psycopg2, "connect", side_effect=fake_connect):
                with mock.patch.object(test_single, "json_to_csv", return_value=None):
                    test_single.run_single_slice_test(use_new_queries=True)

        self.assertEqual(
            len(connect_calls),
            1,
            "musique ladder should reuse one connection across n=1,5,10,20",
        )


if __name__ == "__main__":
    unittest.main()
