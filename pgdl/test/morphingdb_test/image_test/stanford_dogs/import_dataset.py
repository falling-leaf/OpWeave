'''
Author: Anonymous anonymous@example.com
Date: 2024-08-08 13:52:21
LastEditors: Anonymous anonymous@example.com
LastEditTime: 2024-08-09 13:46:49
FilePath: /morphingdb_test/image_test/cifar10/import_dataset.py
Description: This is the default setting, set `customMade`; open koroFileHeader to view the configuration settings: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import psycopg2
import os
# import evadb  # disabled: evadb import functions commented out (creates hash cache dirs)
from morphingdb_test.config import db_config, stanford_dogs_dataset_path

IMAGE_COUNT_LIST = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

IMAGE_TABLE = 'stanford_dogs_image_table'
IMAGE_VECTOR_TABLE = 'stanford_dogs_image_vector_table'

# Initialize an empty list to store all file paths
image_names = []

# Walk IMAGE_PRE_PATH and all its subdirectories with os.walk
for root, dirs, files in os.walk(stanford_dogs_dataset_path):
    for file in files:
        # Build the full file path
        full_path = os.path.join(root, file)
        # Append the file path to the list
        image_names.append(full_path)

image_names = image_names[:10000]






def import_stanford_dogs_dataset():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # Create the table
    cur.execute("create table if not exists " + IMAGE_TABLE  + " (id int, image_path text);")
    conn.commit()

    # Delete rows from the table
    cur.execute("delete from " + IMAGE_TABLE + ";")
    conn.commit()


    # Insert into the table
    for index in range(10000):
        sql = f"INSERT INTO " + IMAGE_TABLE + " (id, image_path) VALUES ({},'{}')".format(index+1, image_names[index])
        cur.execute(sql)
    conn.commit()
    conn.close()


def import_stanford_dogs_vector_dataset():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # Create the table
    cur.execute("create table if not exists " + IMAGE_VECTOR_TABLE  + " (id int, image_vector mvec);")
    conn.commit()

    # Delete rows from the table
    cur.execute("delete from " + IMAGE_VECTOR_TABLE + ";")
    conn.commit()

    for index in range(10000):
        sql = f"INSERT INTO " + IMAGE_VECTOR_TABLE + " (id, image_vector) VALUES ({},{})".format(index+1, "image_to_vector(256,224,0.485,0.456,0.406,0.229,0.224,0.225, '{}')".format(image_names[index]))
        cur.execute(sql)
    conn.commit()
    conn.close()

# def import_evadb_stanford_dogs_dataset():
#     # Connect evadb -- DISABLED: creates hash-named cache dirs (e.g. 9bfebd13...)
#     # cursor = evadb.connect().cursor()
#     # cursor.query("DROP TABLE IF EXISTS STANFORD").df()
#     # for index in range(10000):
#     #     cursor.query("LOAD IMAGE '{}' INTO STANFORD".format(image_names[index])).df()
#     pass