import psycopg2
import time
import json
#from morphingdb_test.config import db_config

db_config = {
    "dbname": "postgres",
    "host": "localhost",
    "port": "5448",
    "user": "lhh"
}


def api_load_model_test():
    # Initialize the results list
    results = []

    # Define the SQL queries to execute
    queries = [
        ("resnet18", "/home/lhh/morphingdb/MorphingDB_test/models/resnet18_vec.pt"),
        ("resnet50", "/home/lhh/models/resnet50_resnet50_nabird.pt"),
        ("alexnet", "/home/lhh/morphingdb/MorphingDB_test/models/alexnet_stanford_dogs.pt"),
        ("googlenet", "/home/lhh/morphingdb/MorphingDB_test/models/googlenet_cifar10.pt")
    ]

    for model_name, model_path in queries:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        start = time.time()
        sql = f"select api_load_model('{model_name}','{model_path}');"
        cur.execute(sql)
        end = time.time()

        # Compute the execution time
        execution_time = end - start

        # Store the result as a dict
        result = {
            "sql": sql,
            "execution_time": execution_time
        }

        # Append the dict to the results list
        results.append(result)

        # Close the database connection
        cur.close()
        conn.close()

    # Write the results list to a JSON file
    with open("result/api_load_model_results.json", "w") as result_file:
        json.dump(results, result_file, indent=4)








if __name__ == "__main__":
    api_load_model_test()