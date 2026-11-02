import psycopg2
import time
import json
import re
import os
from morphingdb_test.config import db_config, financial_phrasebank_model_path
from morphingdb_test.text_test.financial_phrasebank.import_dataset import (
    import_financial_phrasebank_dataset,
    import_financial_phrasebank_mvec_dataset,
    TEXT_TABLE,
    TEXT_VECTOR_TABLE
)


TEXT_COUNT_LIST = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

TEXT_TEST_FILE = 'result/finance_phrasebank_test.json'
TEXT_VECTOR_TEST_FILE = 'result/finance_phrasebank_mvec_test.json'
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

    cur.execute("select * from model_info where model_name = 'finance';")
    res = cur.fetchall()
    if len(res) == 0:
        cur.execute("select create_model('finance', '{}', '', '');".format(financial_phrasebank_model_path))

    conn.commit()
    conn.close()


def financial_phrasebank_init_data():
    import_financial_phrasebank_dataset()
    print("import financial phrasebank dataset done")
    import_financial_phrasebank_mvec_dataset()
    print("import financial phrasebank mvec dataset done")


def finance_test(limit_flag:str, symbol:str = 'cpu'):
    for count in TEXT_COUNT_LIST:
        # Connect to the database
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        # Execute database operations with conn
        cur.execute("select register_process();")
        start = time.time()
        sql = "select predict_batch_float8('finance', '{}', comment_vec) over (rows between current row and 15 following ) from financial_phrasebank_vector_test limit {};".format(symbol, count)
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
        timing_data_vector.append({"sql": sql,
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

def financial_phrasebank_all_test():
    import_financial_phrasebank_dataset()
    print('import financial phrasebank dataset done')
    import_financial_phrasebank_mvec_dataset()
    print('import financial phrasebank mvec dataset done')
    create_model()
    print('create model done')
    finance_test('', 'gpu')
    print('finance test done')

if __name__ == "__main__":
    create_model()
    finance_test('', 'cpu')