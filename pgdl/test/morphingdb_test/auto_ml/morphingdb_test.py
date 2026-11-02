
import psycopg2
import time
import psutil

from morphingdb_test.config import db_config
from morphingdb_test.image_test.cifar10.import_dataset import IMAGE_TABLE

def auto_ml_test():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    cur.execute("SELECT pg_backend_pid();")
    pid = cur.fetchone()[0]

    # Monitor memory usage of the given PID
    process = psutil.Process(pid)

    # Record memory usage at the start
    start_mem = process.memory_info().rss / (1024 * 1024)  # convert to MB

    s_time = time.time()
    cur.execute("select image_classification('{}', 'image_path', 'cifar10', 10);".format(IMAGE_TABLE))
    e_time = time.time()

    # Record memory usage at the end
    end_mem = process.memory_info().rss / (1024 * 1024)  # convert to MB

    # Compute the memory usage delta
    mem_used = end_mem - start_mem


    print("cost time:{}".format(e_time - s_time))
    print("Memory used: {:.2f} MB".format(mem_used))