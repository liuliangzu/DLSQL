from threading import local
import numpy as np
import pandas as pd
import time
import socket
import psycopg2 as p2
from data_to_tensor import FeatureDictionary,DataParser
ip = '172.17.0.2'
port = 5432
user = 'gpadmin'
dbname = 'gpadmin'

def record(content):
    f = open("/data2/ruike/pg/outDB.sql", 'a')
    f.write(content)
    f.close()

def get_result_from_db(sql):
    user = 'gpadmin'
    dbname = 'gpadmin'
    port = 5432
    ip = '172.17.0.2'
    conn = p2.connect(host = ip, user = user, dbname = dbname, port = port)
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    conn.commit()
    return results

def recv_weights(current_seg_id):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 33000+current_seg_id
    server_address = ('172.17.0.3', port)
    client_socket.bind(server_address)
    client_socket.listen(1)
    socket, address = client_socket.accept()
    data = client_socket.recv(1000000)
    recv_list = eval(data.decode())

    client_socket.close()
    socket.close()

def send_weights(weights_list,current_seg_id):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = current_seg_id + 32000
    server_address = ('172.17.0.2', port)
    client_socket.connect(server_address)
    client_socket.send(str(weights_list).encode())
    client_socket.close()

class data_loader:
    def __init__(self):
        record("data_loader constructed!\n")
        return

    def get_data(self, source_table_name, current_seg_id):
        t1 = time.time()
        conn = p2.connect(host = ip, user = user, dbname = dbname, port = port)
        sql = 'select * from {} where gp_segment_id = {}'.format(source_table_name,current_seg_id)
        data_frame = pd.read_sql(sql,conn)
        record("seg {} get data from greenplum last {}\n".format(current_seg_id, time.time() - t1))
        return data_frame

class tensor_transfer:
    def __init__(self):
        record("data to tensor started!\n")
        return
    
    def tensor_tranfer_df(self,dfTrain):
        fd = FeatureDictionary(dfTrain=dfTrain, dfTest=None,
                           numeric_cols=['i1','i2','i3','i4','i5','i6','i7','i8','i9','i10','i11','i12','i13'],
                           ignore_cols=['id'])
        data_parser = DataParser(feat_dict=fd)
        Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
        return Xi_train, Xv_train, y_train, fd.feat_dim, len(Xi_train[0])


class model_weights_transfer:
    def __init__(self):
        return 
    
    def send_to_master(self,weights,current_seg_id):
        send_weights(weights,current_seg_id)
        return

class model_weights_get:
    def __init__(self):
        return