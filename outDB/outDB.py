# use 4 threads or 4 .sh files to run model Deepfm on segment node
# for each thread or file:
    # step1 : building a Model with all parameters
    # step2 : select data buffer from master db
    # step3 : fit with each buffer
    # step4 : average 4 model weights
    # step5 : go to step1
# so we get 5 time
    # 1.building environment
    # 2.get a buffer or block data(for our methods, use buffer is ok)
    # 3.fit time
    # 4.average time
    # 5.all time
import threading
import numpy as np
import pandas as pd
import psycopg2 as p2
import tensorflow as tf
from DeepFM import DeepFM_outDB
import time

#args setting:
epoch_num = 10
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

def np_array_float32(var, var_shape):
    arr = np.frombuffer(var, dtype=np.float32)
    arr.shape = var_shape
    return arr

def np_array_int16(var, var_shape):
    arr = np.frombuffer(var, dtype=np.int16)
    arr.shape = var_shape
    return arr

def tf_deserialize_as_nd_weights(model_weights_serialized, model_shapes):
    """
    The output of this function is used to set keras model weights using the
    function model.set_weights()
    :param model_weights_serialized: bytestring containing model weights
    :param model_shapes: list containing the shapes of each layer.
    :return: list of nd numpy arrays containing all of the
        weights
    """
    if not model_weights_serialized or not model_shapes:
        return None

    i, j, model_weights = 0, 0, []
    model_weights_serialized = np.fromstring(model_weights_serialized, dtype=np.float32)
    total_model_shape =  sum([reduce(lambda x, y: x * y, ls if ls else [1]) for ls  in model_shapes])
    total_weights_shape = model_weights_serialized.size
    assert total_model_shape == total_weights_shape, "Number of elements in model weights({0}) doesn't match model({1})." .format(total_weights_shape, total_model_shape)
    while j < len(model_shapes):
        next_pointer = i + reduce(lambda x, y: x * y, model_shapes[j] if model_shapes[j] else [1])
        weight_arr_portion = model_weights_serialized[i:next_pointer]
        model_weights.append(np.array(weight_arr_portion).reshape(model_shapes[j]))
        i, j = next_pointer, j + 1

    return model_weights

def computing(current_seg_id, first_iter):
    model = DeepFM_outDB(**dfm_params)
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
    source_table = source_table_name
    t4 = time.time()
    record("seg {} start train at {}\n".format(current_seg_id, t4 - t1))
    data_time = 0
    train_time = 0
    for buffer in range(buffer_num):
        buffer_index = buffer * 4 + current_seg_id
        dt = time.time()
        select_query = '''  SELECT xi,xv,y,xi_shape,xv_shape,y_shape
                            FROM {source_table}
                            where {source_table}.buffer_id = {buffer_index} and gp_segment_id = {current_seg_id}'''.format(**locals())
        data = get_result_from_db(select_query)
        xi = list()
        xv = list()
        y = list()
        xi_shape = list()
        xv_shape = list()
        target_shape = list()
        for i, row in enumerate(data):
            xi.append(row[0])
            xv.append(row[1])
            y.append(row[2])
            xi_shape.append(row[3])
            xv_shape.append(row[4])
            target_shape.append(row[5])
        xi_train = np_array_float32(xi[0],xi_shape[0])
        xv_train = np_array_float32(xv[0],xv_shape[0])
        y_train = np_array_int16(y[0],target_shape[0])
        xi_tmp = xi_train.T
        xv_tmp = xv_train.T
        xi = [np.array(xi_tmp[i, :]) for i in range(xi_tmp.shape[0])]
        xv = [np.array(xv_tmp[i, :]) for i in range(xv_tmp.shape[0])]
        Xi_train = np.array(xi).T
        Xv_train = np.array(xv).T
        Y_train = list()
        for y in y_train:
            Y_train.append(y[0])
        Y_train = np.array(Y_train).reshape(-1,1)
        dt_last = time.time() - dt
        data_time = data_time + dt_last
        tt = time.time()
        model.train_on_buffer(Xi_train,Xv_train,Y_train)
        tt_last = time.time() - tt
        train_time = train_time + tt_last
    t5 = time.time()
    record("seg {} end train last {}, data_time {}, train_time {}\n".format(current_seg_id, t5 - t4, data_time, train_time))
    record("seg {} start save weights at {}\n".format(current_seg_id, t5 - t1))
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
        thread = threading.Thread(target=computing, args=(seg_id, first_iter),name='training_{}'.format(seg_id))
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