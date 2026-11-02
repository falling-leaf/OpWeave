ROOT=/opt/opweave
PG=$ROOT/pg_base

cd "$ROOT"
"$PG/bin/pg_ctl" -D "$PG/data" stop -m immediate
"$PG/bin/pg_ctl" -D "$PG/data" start
"$PG/bin/psql" -d postgres
