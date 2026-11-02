import evadb
import time
import json
import os
from morphingdb_test.image_test.stanford_dogs.import_dataset import import_evadb_stanford_dogs_dataset

IMAGE_COUNT_LIST = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
IMAGE_TEST_FILE = 'result/evadb_stanford_dogs_test.json'

cursor = evadb.connect().cursor()



query = cursor.query("""
    CREATE FUNCTION IF NOT EXISTS AlexnetStanford
    INPUT  (data NDARRAY FLOAT32(ANYDIM))
    OUTPUT (labels NDARRAY FLOAT32(10))
    TYPE  Classification
    IMPL  './morphingdb_test/image_test/stanford_dogs/evadb_alexnet_stanford_dog.py';
""").df()


def gpu_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    for count in IMAGE_COUNT_LIST:
        start_time = time.time()
        sql = """
            SELECT AlexnetStanford(data)
            FROM STANFORD limit {};
        """.format(count)
        res = cursor.query(sql).df()
        end_time = time.time()
        print("cost time:", end_time - start_time, "s") 

        try:
            with open(IMAGE_TEST_FILE, 'r') as f_image:
                # Load existing data
                timing_data_image = json.load(f_image)

            # Iterate over the list
            for item in timing_data_image:
                # Find the entry with count 100 and zero total_time and scan_time
                print(item)
                if item["count"] == count and item["total_time"] == 0 and item["scan_time"] == 0:
                    # Update total_time and scan_time
                    item["total_time"] = end_time - start_time
                    item["scan_time"] = (end_time - start_time) - item['load_model_time'] - item['pre_time'] - item['infer_time'] - item['post_time']
                    item["sql"] = "gpu" + sql
                    # The needed entry is found, so the loop can end
                    break
            
            print(timing_data_image)
            # Write the updated list back to the file
            with open(IMAGE_TEST_FILE, 'w') as f_image:
                json.dump(timing_data_image, f_image, indent=4)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"An error occurred: {e}")


def cpu_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    for count in IMAGE_COUNT_LIST:
        start_time = time.time()
        sql = """
            SELECT AlexnetStanford(data)
            FROM STANFORD limit {};
        """.format(count)
        res = cursor.query(sql).df()
        end_time = time.time()
        print("cost time:", end_time - start_time, "s") 

        try:
            with open(IMAGE_TEST_FILE, 'r') as f_image:
                # Load existing data
                timing_data_image = json.load(f_image)

            # Iterate over the list
            for item in timing_data_image:
                # Find the entry with count 100 and zero total_time and scan_time
                print(item)
                if item["count"] == count and item["total_time"] == 0 and item["scan_time"] == 0:
                    # Update total_time and scan_time
                    item["total_time"] = end_time - start_time
                    item["scan_time"] = (end_time - start_time) - item['load_model_time'] - item['pre_time'] - item['infer_time'] - item['post_time']
                    # The needed entry is found, so the loop can end
                    item["sql"] = "cpu" + sql
                    break
            
            print(timing_data_image)
            # Write the updated list back to the file
            with open(IMAGE_TEST_FILE, 'w') as f_image:
                json.dump(timing_data_image, f_image, indent=4)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"An error occurred: {e}")


def evadb_stanford_dogs_test():
    import_evadb_stanford_dogs_dataset()
    gpu_test()
    #cpu_test()

if __name__ == "__main__":
    gpu_test()
    cpu_test()