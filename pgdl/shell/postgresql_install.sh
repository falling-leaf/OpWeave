cd /opt/opweave/pgdl/third_party/postgresql-13.22/
make distclean
./configure --prefix=/opt/opweave/pg_base
make -j$(nproc)
make install
/opt/opweave/pg_base/bin/initdb -D /opt/opweave/pg_base/data
