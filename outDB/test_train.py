from re import T
import threading
import numpy as np
import pandas as pd
import psycopg2 as p2
import tensorflow as tf
from DeepFM import DeepFM_outDB
import time
from interact import data_loader, model_weights_transfer,tensor_transfer,model_weights_get,model_weights_transfer

#args setting:
epoch_num = 1
batch_size = 1024
deep_layer = 400
feature_num = 33000000
embedding_fields = 4
colums = 39
source_table_name = 'criteo_tensor_packed'
buffer_num = 900

dfm_params = {
        "use_fm": True,
        "use_deep": True,
        "feature_size":33000000,
        "field_size": 39,
        "embedding_size": 4,
        "dropout_fm": [1.0, 1.0],
        "deep_layers": [400, 400],
        "dropout_deep": [0.5, 0.5, 0.5],
        "deep_layers_activation": tf.nn.relu,
        "epoch": 30,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer_type": "gd",
        "batch_norm": 0,
        "batch_norm_decay": 0.995,
        "l2_reg": 0.01,
        "verbose": True,
        "random_seed": 2017
    }

def record(content):
    f = open("/data2/ruike/pg/outDB.sql", 'a')
    f.write(content)
    f.close()

def save_flatten_weights(content,seg_id):
    f = open("./weights_{}.txt".format(seg_id), 'a')
    f.write(content)
    f.close()

def read_flatten_weights(seg_id):
    f = open("./weights_{}.txt".format(seg_id), 'rw')
    res = f.read()
    f.truncate(0)
    f.close()
    return res

def model_train(current_seg_id,first_iter):
    t1 = time.time()
    record("seg {} start computing at {}\n".format(current_seg_id, t1))
    if first_iter == 1:
        t2 = time.time()
        record("seg {} start set_weghts at {}\n".format(current_seg_id, t2 - t1))
        weights = read_flatten_weights(999)
        variables = model.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        weights_updated = [model.sess.run(v) for v in variables]
        shapes = [v.shape for v in weights_updated]
        model_weights = tf_deserialize_as_nd_weights(weights,shapes)
        model.set_model_weights(model_weights)
        t3 = time.time()
        record("seg {} end set_weghts last {}\n".format(current_seg_id, t3 - t2))
    dl = data_loader()
    dftrain = dl.get_data('criteo_train_data',current_seg_id)
    ttf= tensor_transfer()
    xi,xv,y,feature_num,field_size = ttf.tensor_tranfer_df(dfTrain=dftrain)
    dfm_params["field_size"] = field_size
    dfm_params["feature_size"] = feature_num
    t4 = time.time()
    record("seg {} start train at {}\n".format(current_seg_id, t4 - t1))
    model = DeepFM_outDB(**dfm_params)
    model.train_on_buffer(xi,xv,np.array(y).reshape(-1,1))
    t5 = time.time()
    record("seg {} end train last {}\n".format(current_seg_id, t5 - t4))
    variables = model.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    variables_value = [model.sess.run(v) for v in variables]
    flattened_weights = [np.float32(w).tostring() for w in variables_value]
    flattened_weights = "".join(flattened_weights)
    save_flatten_weights(flattened_weights, current_seg_id)
    t6 = time.time()
    record("seg {} end save weights last {}\n".format(current_seg_id, t6 - t5))
    return 

def model_average():
    weights = list()
    for i in range(4):
        tmp = read_flatten_weights(i)
        return_state = np.fromstring(tmp, dtype=np.float32)
        weights.append(return_state)
    weights_avg = np.array(weights).mean(axis = 0)
    weights_save = np.float32(weights_avg).tostring()
    weights_save = "".join(weights_save)
    save_flatten_weights(weights_save, 999)

first_iter = 0
for i in range(epoch_num):
    threading_dict = {}
    if first_iter == 1:
        model_average()
    for seg_id in range(4):
        thread = threading.Thread(target=model_train, args=(seg_id, first_iter),name='training_{}'.format(seg_id))
        threading_dict[seg_id] = thread
        start_or_not = True #memory_check(ip, port, sql_mem_use)
        record("memory for seg {} is {}\n".format(seg_id, start_or_not))
        if start_or_not:
            try :
                thread.start()
            except:
                record("thread {} start with error!\n".format(seg_id))
    for thread_id, thread in threading_dict.items():
        thread.join()
        del threading_dict[thread_id]
    first_iter = 1
    record("*"*50 + "\n")
    record("epoch {} end;\n".format(i))