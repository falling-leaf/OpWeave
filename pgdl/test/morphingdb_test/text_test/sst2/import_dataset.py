import psycopg2
import os
import pandas as pd
from morphingdb_test.config import db_config, sst2_dataset_path, spiece_model_path

TEXT_COUNT_LIST = [1000]

TEXT_TABLE = 'nlp_test'

IMAGE_VECTOR_TABLE = 'nlp_vector_test'

df = pd.read_csv(sst2_dataset_path, sep='\t')



def import_sst2_mvec_dataset():
    # Connect to the database
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # Create the table
    cur.execute("create table if not exists " + IMAGE_VECTOR_TABLE + " (comment_vec mvec);")
    conn.commit()

    # Delete rows from the table
    cur.execute("delete from " + IMAGE_VECTOR_TABLE + ";")
    conn.commit()

    inserted_count = 0
    # Insert into the table
    for index, row in df.iterrows():
        if inserted_count >= 10001:  # check whether 10000 rows have been inserted
            break  # exit the loop once 10000 rows have been inserted
        sql_vec = f"INSERT INTO " + IMAGE_VECTOR_TABLE + " (comment_vec) VALUES ({})".format("text_to_vector('{}', '{}')".format(spiece_model_path, row['sentence'].replace("'","''")))
        cur.execute(sql_vec)
        conn.commit()
        inserted_count += 1
    conn.close()


def import_sst2_dataset():
    # Connect to the database
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # Create the table
    cur.execute("create table if not exists " + TEXT_TABLE + " (comment text);")
    conn.commit()

    # Delete rows from the table
    cur.execute("delete from " + TEXT_TABLE + ";")
    conn.commit()

    inserted_count = 0
    # Insert into the table
    for index, row in df.iterrows():
        if inserted_count >= 10001:  # check whether 10000 rows have been inserted
            break  # exit the loop once 10000 rows have been inserted
        sql_comment = f"INSERT INTO " + TEXT_TABLE + " (comment) VALUES ('{}')".format(row['sentence'].replace("'","''"))
        cur.execute(sql_comment)
        conn.commit()
        inserted_count += 1
    conn.close()