import pandas as pd
import psycopg2
import os
from morphingdb_test.config import db_config, year_predict_dataset_path


# db_config = {
#     "dbname": "postgres",
#     "host": "localhost",
#     "port": "5432",
#     "user": "postgres",
#     "password": "123456"
# }


def import_year_predict_mvec_table():
    # Connect to the database
    dataframe = pd.read_csv(year_predict_dataset_path)

    # Select all columns starting with 'value'
    value_columns = dataframe.columns[1:91]

    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    cur.execute("drop table if exists year_predict_test;")
    cur.execute("create table year_predict_test ( \
                    data mvec, \
                    res float4 \
                );")

    
    # Iterate over each row of the DataFrame and assemble the value columns into a list-style string
    formatted_rows = []
    for index, row in dataframe.iterrows():
        # Extract the value column value from the current row
        values = row[value_columns].tolist()
        # Convert the list of values to a string
        values_str = '[' + ', '.join(str(value) for value in values) + ']'
        # Append the result to the list
        cur.execute("insert into year_predict_test values('{}', {} );".format(values_str+'{1,90}', row['value0']))
        if(index % 10 == 0):
            conn.commit()
    
    conn.close()


def import_year_predict_table():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    ## origin table test
    dataframe = pd.read_csv(year_predict_dataset_path)
    value_columns = dataframe.columns.tolist()
    
    data_types = ['float4' for _ in value_columns] 

    cur.execute("drop table if exists year_predict_origin_test;")
    create_table_sql = 'create table year_predict_origin_test (\n  '
    for i, column in enumerate(value_columns):
        create_table_sql += f"\"{column}\" {data_types[i]},\n"
    create_table_sql = create_table_sql.rstrip(',\n') + '\n);'
    cur.execute(create_table_sql)
 
    cur.execute("copy year_predict_origin_test FROM '{}' WITH (FORMAT csv, HEADER true, DELIMITER ',');".format(year_predict_dataset_path))
    conn.commit()

    conn.close()