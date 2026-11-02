import psycopg2
import os
# import evadb  # disabled: evadb import functions commented out (creates hash cache dirs)
from morphingdb_test.config import db_config, cifar10_dataset_path

IMAGE_COUNT_LIST = [1000]

IMAGE_TABLE = 'cifar_image_table'
IMAGE_VECTOR_TABLE = 'cifar_image_vector_table'

# Initialize an empty list to store all file paths
image_names = []

# Walk IMAGE_PRE_PATH and all its subdirectories with os.walk
for root, dirs, files in os.walk(cifar10_dataset_path):
    for file in files:
        # Build the full file path
        full_path = os.path.join(root, file)
        # Append the file path to the list
        image_names.append(full_path)



def cifar10_import_mvec_dataset():
    # Connect to the MorphingDB database
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute("create table if not exists " + IMAGE_VECTOR_TABLE  + " (id int, image_vector mvec);")
    conn.commit()

    cur.execute("delete from " + IMAGE_VECTOR_TABLE + ";")
    conn.commit()

    for index in range(len(image_names)):
        sql = f"INSERT INTO " + IMAGE_VECTOR_TABLE + " (id, image_vector) VALUES ({},{})".format(index+1, "image_to_vector(224,224,0.4914,0.4822,0.4465,0.2023,0.1994,0.2010, '{}')".format(image_names[index]))
        #print(sql)
        cur.execute(sql)
    conn.commit()
    conn.close()


def cifar10_import_url_dataset():
    # Connect to the MorphingDB database
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute("create table if not exists " + IMAGE_TABLE  + " (id int, image_path text);")
    conn.commit()

    cur.execute("delete from " + IMAGE_TABLE + ";")
    conn.commit()

    for index in range(len(image_names)):
        sql = f"INSERT INTO " + IMAGE_TABLE + " (id, image_path) VALUES ({},'{}')".format(index+1, image_names[index])
        #print(sql)
        cur.execute(sql)
    conn.commit()
    conn.close()


# def cifar10_import_evadb_dataset():
#     # Connect evadb -- DISABLED: creates hash-named cache dirs (e.g. 9bfebd13...)
#     # cursor = evadb.connect().cursor()
#     # cursor.query("DROP TABLE IF EXISTS CIFAR10").df()
#     # for index in range(len(image_names)):
#     #     cursor.query("LOAD IMAGE '{}' INTO CIFAR10".format(image_names[index])).df()
#     pass