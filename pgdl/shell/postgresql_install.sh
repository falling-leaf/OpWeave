ROOT=/opt/opweave
PG=$ROOT/pg_base

cd "$ROOT/pgdl/third_party/postgresql-13.22/"
make distclean
./configure --prefix="$PG"
make -j"$(nproc)"
make install
"$PG/bin/initdb" -D "$PG/data"
