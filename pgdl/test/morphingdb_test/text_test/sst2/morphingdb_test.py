from venv import create
import psycopg2
import time
import json
import re
import os
from morphingdb_test.config import db_config, sst2_model_path
from morphingdb_test.text_test.sst2.import_dataset import (
    import_sst2_dataset,
    import_sst2_mvec_dataset,
    TEXT_TABLE,
    IMAGE_VECTOR_TABLE
)


TEXT_COUNT_LIST = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

TEXT_TEST_FILE = 'result/sst2_test.json'
TEXT_VECTOR_TEST_FILE = 'result/sst2_vector_test.json'
timing_data_image = []
timing_data_vector = []

# parse time
def parse_timing_info(timing_str):
    # Match the required timing information with a regex
    pattern = r'total: (\d+) ms\n load model: (\d+) ms\((\d+\.\d+)%\)\n pre process: (\d+) ms\((\d+\.\d+)%\)\n infer: (\d+) ms\((\d+\.\d+)%\)\n post process: (\d+) ms\((\d+\.\d+)%\)'
    
    # Use search to find the substring matching the regex
    match = re.search(pattern, timing_str)
    
    if match:
        # Convert the captured groups into a dict
        timing_info = {
            'total_time': int(match.group(1)),
            'load_model_time': int(match.group(2)),
            'load_model_percent': float(match.group(3)),
            'pre_time': int(match.group(4)),
            'pre_percent': float(match.group(5)),
            'infer_time': int(match.group(6)),
            'infer_percent': float(match.group(7)),
            'post_time': int(match.group(8)),
            'post_percent': float(match.group(9)),
        }
        return timing_info
    else:
        # If no match is found, return None or raise an exception
        return None

# Paths are stored in the database
# for count in TEXT_COUNT_LIST:
#     # Connect to the database
#     conn = psycopg2.connect(**db_config)
#     cur = conn.cursor()
#     # Execute database operations with conn
#     cur.execute("select register_process();")
#     start = time.time()
#     cur.execute("select predict_batch_float8('sst2', 'cpu', comment) over (rows between current row and 15 following ) from nlp_test limit {};".format(count))
#     end = time.time()
#     cur.execute("select print_cost();")
#     res = parse_timing_info(cur.fetchall()[0][0])
#     res['scan_time'] = (end - start)*1000000 - res['total_time']
#     res['total_time'] = (end - start)*1000000
#     conn.close()


#     # Read and append to TEXT_TEST_FILE
#     try:
#         with open(TEXT_TEST_FILE, 'r') as f_image:
#             # Try to load existing data
#             timing_data_image = json.load(f_image)
#     except (FileNotFoundError, json.JSONDecodeError):
#         # If the file does not exist or is not valid JSON, initialize it as an empty list
#         timing_data_image = []

#     # Append the new records to the list
#     timing_data_image.append({"count": count, 
#                               "total_time": res["total_time"]/1000000, 
#                               "scan_time": res["scan_time"]/1000000,
#                               "load_model_time": res["load_model_time"]/1000000, 
#                               "pre_time": res["pre_time"]/1000000,
#                               "infer_time": res["infer_time"]/1000000, 
#                               "post_time": res["post_time"]/1000000})

#     print(timing_data_image)
#     # Write back to the file
#     with open(TEXT_TEST_FILE, 'w') as f_image:
#         json.dump(timing_data_image, f_image, indent=4)

def create_model():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    cur.execute("select * from model_info where model_name = 'sst2_vec';")
    res = cur.fetchall()
    if len(res) == 0:
        cur.execute("select create_model('sst2_vec', '{}', '', '');".format(sst2_model_path))

    conn.commit()
    conn.close()


def sst2_init_data():
    import_sst2_dataset()
    print("import sst2 dataset done")
    import_sst2_mvec_dataset()
    print("import sst2 mvec dataset done")


def sst2_vec_test(limit_flag:str, symbol:str = 'cpu'):
    for count in TEXT_COUNT_LIST:
        # Connect to the database
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        # Execute database operations with conn
        cur.execute("select register_process();")
        start = time.time()
        sql = "select predict_batch_float8('sst2_vec', '{}', comment_vec) over (rows between current row and 15 following ) from {} limit {};".format(symbol, IMAGE_VECTOR_TABLE, count)
        cur.execute(sql)
        end = time.time()
        cur.execute("select print_cost();")
        res = parse_timing_info(cur.fetchall()[0][0])
        res['scan_time'] = (end - start)*1000000 - res['total_time']
        res['total_time'] = (end - start)*1000000
        conn.close()


        # Read and append to TEXT_VECTOR_TEST_FILE
        try:
            with open(TEXT_VECTOR_TEST_FILE.format(limit_flag), 'r') as f_vector:
                # Try to load existing data
                timing_data_vector = json.load(f_vector)
        except (FileNotFoundError, json.JSONDecodeError):
            # If the file does not exist or is not valid JSON, initialize it as an empty list
            timing_data_vector = []

        # Append the new records to the list
        timing_data_vector.append({ "sql": sql,
                                "count": count, 
                                "total_time": res["total_time"]/1000000, 
                                "scan_time": res["scan_time"]/1000000,
                                "load_model_time": res["load_model_time"]/1000000, 
                                "pre_time": res["pre_time"]/1000000,
                                "infer_time": res["infer_time"]/1000000, 
                                "post_time": res["post_time"]/1000000})

        # Write back to the file
        with open(TEXT_VECTOR_TEST_FILE.format(limit_flag), 'w') as f_vector:
            json.dump(timing_data_vector, f_vector, indent=4)

def sst2_all_test():
    import_sst2_dataset()
    print('import sst2 dataset done')
    import_sst2_mvec_dataset()
    print('import sst2 mvec dataset done')
    create_model()
    print('create model done')
    sst2_vec_test('', 'cpu')
    print('sst2 vec test done')


if __name__ == "__main__":
    create_model()
    sst2_vec_test('mem_2', 'cpu')
    #sst2_vec_test('mem_2', 'gpu')