import evadb
import time
import json
import os
from morphingdb_test.config import evadb_db_config

TEXT_COUNT_LIST = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
TEXT_TEST_FILE = 'result/evadb_imdb_test.json'

cursor = evadb.connect().cursor()

query = cursor.query("""
    DROP DATABASE IF EXISTS postgres_data;
""").df()
query = cursor.query(f"""
    CREATE DATABASE IF NOT EXISTS postgres_data
    WITH ENGINE = 'postgres',
    PARAMETERS = {evadb_db_config}
""").df()

# query = cursor.query("""
#     DROP FUNCTION IMDBTest;
# """).df()

query = cursor.query("""
    CREATE FUNCTION IF NOT EXISTS IMDBTest
    INPUT  (question TEXT)
    OUTPUT (labels NDARRAY FLOAT32(3))
    TYPE  Classification
    IMPL  './morphingdb_test/text_test/imdb/evadb_imdb.py';
""").df()


def gpu_test():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for count in TEXT_COUNT_LIST:
        start_time = time.time()
        sql = "SELECT IMDBTest(comment) FROM postgres_data.imdb_test limit {};".format(count)
        res = cursor.query(sql).df()
        print(res)
        end_time = time.time()
        print("cost time:", end_time - start_time, "s") 

        try:
            with open(TEXT_TEST_FILE, 'r') as f_image:
                # Load existing data
                timing_data_image = json.load(f_image)

            # Iterate over the list
            for item in timing_data_image:
                # Find the entry with count 100 and zero total_time and scan_time
                if item["count"] == count-1 and item["total_time"] == 0 and item["scan_time"] == 0:
                    # Update total_time and scan_time
                    item["sql"] = sql
                    item["total_time"] = end_time - start_time
                    item["scan_time"] = (end_time - start_time) - item['load_model_time'] - item['pre_time'] - item['infer_time'] - item['post_time']
                    # The needed entry is found, so the loop can end
                    break
            
            print("timing_data_image", timing_data_image)
            # Write the updated list back to the file
            with open(TEXT_TEST_FILE, 'w') as f_image:
                json.dump(timing_data_image, f_image, indent=4)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"An error occurred: {e}")

def cpu_test():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for count in TEXT_COUNT_LIST:
        start_time = time.time()
        sql = "SELECT IMDBTest(comment) FROM postgres_data.imdb_test limit {};".format(count)
        res = cursor.query(sql).df()
        print(res)
        end_time = time.time()
        print("cost time:", end_time - start_time, "s") 

        try:
            with open(TEXT_TEST_FILE, 'r') as f_image:
                # Load existing data
                timing_data_image = json.load(f_image)

            # Iterate over the list
            for item in timing_data_image:
                # Find the entry with count 100 and zero total_time and scan_time
                if item["count"] == count and item["total_time"] == 0 and item["scan_time"] == 0:
                    # Update total_time and scan_time
                    item["sql"] = sql
                    item["total_time"] = end_time - start_time
                    item["scan_time"] = (end_time - start_time) - item['load_model_time'] - item['pre_time'] - item['infer_time'] - item['post_time']
                    # The needed entry is found, so the loop can end
                    break
            
            print("timing_data_image", timing_data_image)
            # Write the updated list back to the file
            with open(TEXT_TEST_FILE, 'w') as f_image:
                json.dump(timing_data_image, f_image, indent=4)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"An error occurred: {e}")


def evadb_imdb_test():
    #gpu_test()
    cpu_test()

if __name__ == "__main__":
    gpu_test()
    cpu_test()