import psycopg2
import time
import json
import re
import os
from morphingdb_test.config import db_config, imagenet_model_path


BATCH_SIZE = [2,4,8,16,32,64,128,256,512]
IMAGE_COUNT_LIST = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
IMAGE_TABLE = 'imagenet_image_table'
IMAGE_VECTOR_TABLE = 'imagenet_image_vector_table'

# IMAGE_TEST_FILE = 'result/morphingdb_batch_test.json'
IMAGE_VECTOR_TEST_FILE = 'result/batch_vector_test.json'
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
    

def create_model():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    cur.execute("select * from model_info where model_name = 'defect_vec';")
    res = cur.fetchall()
    if len(res) == 0:
        cur.execute("select create_model('defect_vec', '{}', '', '');".format(imagenet_model_path))
    conn.commit()

    cur.execute("select * from model_info where model_name = 'defect';")
    res = cur.fetchall()
    if len(res) == 0:
        cur.execute("select create_model('defect', '{}', '', '');".format(imagenet_model_path))
    conn.commit()
    conn.close()



def image_batch_test(limit_flag:str, symbol:str = 'cpu'):
    for batch_size in BATCH_SIZE:
        # for count in IMAGE_COUNT_LIST:
        #     # Connect to the database
        #     conn = psycopg2.connect(**db_config)
        #     cur = conn.cursor()
        #     # Execute database operations with conn
        #     cur.execute("select register_process();")
        #     start = time.time()
        #     #print("select predict_batch_float8('defect', 'cpu', image_path) over (rows between current row and 15 following) from image_table_10000 limit {};".format(count))
        #     sql = "select predict_batch_float8('defect', 'gpu', image_path) over (rows between current row and {} following) from {} limit {};".format(batch_size-1, IMAGE_TABLE, count)
        #     cur.execute(sql)
        #     end = time.time()
        #     cur.execute("select print_cost();")
        #     res = parse_timing_info(cur.fetchall()[0][0])
        #     res['scan_time'] = (end - start)*1000000 - res['total_time']
        #     res['total_time'] = (end - start)*1000000
        #     conn.close()


        #     # Read and append to IMAGE_TEST_FILE
        #     try:
        #         with open(IMAGE_TEST_FILE.format(batch_size), 'r') as f_image:
        #             # Try to load existing data
        #             timing_data_image = json.load(f_image)
        #     except (FileNotFoundError, json.JSONDecodeError):
        #         # If the file does not exist or is not valid JSON, initialize it as an empty list
        #         timing_data_image = []

        #     # Append the new records to the list
        #     timing_data_image.append({"sql": sql,
        #                             "count": count, 
        #                             "total_time": res["total_time"]/1000000, 
        #                             "scan_time": res["scan_time"]/1000000,
        #                             "load_model_time": res["load_model_time"]/1000000, 
        #                             "pre_time": res["pre_time"]/1000000,
        #                             "infer_time": res["infer_time"]/1000000, 
        #                             "post_time": res["post_time"]/1000000})

        #     # Write back to the file
        #     with open(IMAGE_TEST_FILE.format(batch_size), 'w') as f_image:
        #         json.dump(timing_data_image, f_image, indent=4)
        # Vectors are stored in the database
        for count in IMAGE_COUNT_LIST:
            # Connect to the database
            conn = psycopg2.connect(**db_config)
            cur = conn.cursor()
            cur.execute("select register_process();")
            start = time.time()
            sql = "select predict_batch_float8('defect_vec', '{}', image_vector) over (rows between current row and {} following) from {} limit {};".format(symbol, batch_size-1, IMAGE_VECTOR_TABLE, count)
            cur.execute(sql)
            end = time.time()
            cur.execute("select print_cost();")
            res = parse_timing_info(cur.fetchall()[0][0])
            res['scan_time'] = (end - start)*1000000 - res['total_time']
            res['total_time'] = (end - start)*1000000
            conn.close()


            # Read and append to IMAGE_VECTOR_TEST_FILE
            try:
                with open(IMAGE_VECTOR_TEST_FILE.format(batch_size, limit_flag), 'r') as f_vector:
                    # Try to load existing data
                    timing_data_vector = json.load(f_vector)
            except (FileNotFoundError, json.JSONDecodeError):
                # If the file does not exist or is not valid JSON, initialize it as an empty list
                timing_data_vector = []

            # Append the new records to the list
            timing_data_vector.append({"sql": sql,
                                    "count": count, 
                                    "total_time": res["total_time"]/1000000, 
                                    "scan_time": res["scan_time"]/1000000,
                                    "load_model_time": res["load_model_time"]/1000000, 
                                    "pre_time": res["pre_time"]/1000000,
                                    "infer_time": res["infer_time"]/1000000, 
                                    "post_time": res["post_time"]/1000000})


            with open(IMAGE_VECTOR_TEST_FILE.format(batch_size, limit_flag), 'w') as f_vector:
                json.dump(timing_data_vector, f_vector, indent=4)

def batch_all_test():
    create_model()
    image_batch_test('', 'gpu')



if __name__ == "__main__":
    create_model()
    image_batch_test('mem_2', 'cpu')