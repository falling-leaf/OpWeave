cd /opt/opweave
/opt/opweave/pg_base/bin/pg_ctl -D /opt/opweave/pg_base/data stop -m immediate
/opt/opweave/pg_base/bin/pg_ctl -D /opt/opweave/pg_base/data start
/opt/opweave/pg_base/bin/psql -d postgres
