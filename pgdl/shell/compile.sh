ROOT=/opt/opweave
PG=$ROOT/pg_base
SRC=$ROOT/pgdl
# make install keeps the absolute destination path, so DESTDIR=$PG stages into $PG$PG
STAGE=$PG$PG

cd "$ROOT"
"$PG/bin/pg_ctl" -D "$PG/data" stop 2>/dev/null
cd "$SRC"
rm -rf build
mkdir build
cd "$SRC/build"
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
cmake -DCMAKE_PREFIX_PATH="$PG/;$SRC/third_party/libtorch/;$SRC/third_party/onnxruntime-linux-x64-1.18.1/" ..
make -j"$(nproc)"
sudo make install DESTDIR="$PG"
cp -f "$STAGE/lib/postgresql/pgdl.so" "$PG/lib/postgresql/"
cp -rf "$STAGE/lib/postgresql/pgdl" "$PG/lib/postgresql/"
cp -f "$STAGE/share/postgresql/extension/"* "$PG/share/postgresql/extension/"
