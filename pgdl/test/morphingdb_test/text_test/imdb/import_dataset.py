import psycopg2
import os
import pandas as pd
from morphingdb_test.config import db_config, imdb_dataset_path, spiece_model_path

TEXT_COUNT_LIST = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

TEXT_TABLE = 'imdb_test'
IMAGE_VECTOR_TABLE = 'imdb_vector_test'


df = pd.read_parquet(imdb_dataset_path)



def import_imdb_mvec_dataset():
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
    #Insert into the table
    for index, row in df.iterrows():
        inserted_count += 1
        if inserted_count >= 10000:  # check whether 10000 rows have been inserted
            break  # exit the loop once 10000 rows have been inserted
        sql_vec = f"INSERT INTO " + IMAGE_VECTOR_TABLE + " (comment_vec) VALUES ({})".format("text_to_vector('{}','{}')".format(spiece_model_path, df.iloc[index,0].replace("'","''")))
        cur.execute(sql_vec)
        conn.commit()

    conn.close()


def import_imdb_dataset():
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
    #Insert into the table
    for index, row in df.iterrows():
        inserted_count += 1
        if inserted_count >= 10000:  # check whether 10000 rows have been inserted
            break  # exit the loop once 10000 rows have been inserted
        sql_comment = f"INSERT INTO " + TEXT_TABLE + " (comment) VALUES ('{}')".format(row['text'].replace("'","''"))
        cur.execute(sql_comment)
        conn.commit()

    conn.close()

