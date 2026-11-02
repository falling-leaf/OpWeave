cd /opt/opweave
/opt/opweave/pg_base/bin/pg_ctl -D /opt/opweave/pg_base/data stop 2>/dev/null
cd /opt/opweave/pgdl
rm -rf build
mkdir build
cd /opt/opweave/pgdl/build
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
cmake -DCMAKE_PREFIX_PATH="/opt/opweave/pg_base/;/opt/opweave/pgdl/third_party/libtorch/;/opt/opweave/pgdl/third_party/onnxruntime-linux-x64-1.18.1/" ..
make -j$(nproc)
sudo make install DESTDIR=/opt/opweave/pg_base
cp -f /opt/opweave/pg_base/opt/opweave/pg_base/lib/postgresql/pgdl.so /opt/opweave/pg_base/lib/postgresql/
cp -rf /opt/opweave/pg_base/opt/opweave/pg_base/lib/postgresql/pgdl /opt/opweave/pg_base/lib/postgresql/
cp -f /opt/opweave/pg_base/opt/opweave/pg_base/share/postgresql/extension/* /opt/opweave/pg_base/share/postgresql/extension/
