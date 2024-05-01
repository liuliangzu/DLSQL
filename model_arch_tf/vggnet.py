"""
Derived from: https://www.cs.toronto.edu/~frossard/vgg16/vgg16.py
"""
#from tkinter.messagebox import NO
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.datasets import cifar10
import psycopg2 as p2
import sys
import ast
from sklearn.metrics import accuracy_score,roc_auc_score
def record(content):
    f = open("/data2/ruike/pg/madlib_model.sql", 'a')
    f.write(content)
    f.close()

def tf_serialize_nd_weights(model_weights):
    """
    This function is called for passing the initial model weights from the keras
    fit function to the keras fit transition function.
    :param model_weights: a list of numpy arrays, what you get from
        keras.get_weights()
    :return: Model weights serialized into a byte string format
    """

    if model_weights is None:
        return None
    flattened_weights = [np.float32(w).tostring() for w in model_weights]
    flattened_weights = "".join(flattened_weights)

    return flattened_weights


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
    model_shapes = eval(model_shapes)
    if isinstance(model_shapes,list):
        #record("{}\n".format(model_shapes))
        total_model_shape = 1
        for x in model_shapes:
            total_model_shape = x * total_model_shape
    else:
        total_model_shape = model_shapes
    total_weights_shape = model_weights_serialized.size

    assert total_model_shape == total_weights_shape, "Number of elements in model weights({0}) doesn't match model({1})." .format(total_weights_shape, total_model_shape)
    model_weights = model_weights_serialized.reshape(model_shapes)

    return model_weights

#524288 1024 4194304 4096 4096000 1000
class VggNetModel_Master(object):

    def __init__(self, num_classes=18, learning_rate=0.01, dropout_keep_prob=0.5):
        self.host = '127.0.0.1'
        self.user = 'gpadmin'
        self.dbname = 'gpadmin'
        self.port = 5432
        self.random_seed = 2016
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dense_shape = 1024
        self.dropout_keep_prob = dropout_keep_prob
        self._init_graph()
        self.init_val()
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.master_layer_num = len(variables)
        #cnn_weights = self.graph.get_collection('cnn_weights')
        self.dense_update_placehoders = list()
        self.dense_update_ops = list()
        self.cnn_update_placehoders = list()
        self.cnn_update_ops = list()
        with self.graph.as_default():
            for i, variable_ in enumerate(variables):
                dense_placehoder_temp = tf.placeholder(variable_.dtype)
                self.dense_update_placehoders.append(dense_placehoder_temp)
                self.dense_update_ops.append(tf.assign(variable_, dense_placehoder_temp, validate_shape=False))
            '''for i, variable_ in enumerate(cnn_weights):
                cnn_placehoder_temp = tf.placeholder(variable_.dtype)
                self.cnn_update_placehoders.append(cnn_placehoder_temp)
                self.cnn_update_ops.append(tf.assign(variable_, cnn_placehoder_temp, validate_shape=False))'''

        self.register_model()
        #self.cnn_weights_load()
        
    def optimize(self, learning_rate, train_layers=[]):
        var_list = [v for v in tf.trainable_variables()]
        return tf.train.AdamOptimizer(learning_rate).minimize(self.ploss, var_list=var_list)
        #return tf.train.GradientDescentOptimizer(learning_rate).minimize(self.ploss, var_list=var_list)

    def load_original_weights(self, skip_layers=[]):
        weights = np.load('vgg16_weights.npz')
        keys = sorted(weights.keys())
        var = list()
        for key in keys:
            var.append(weights[key])
        feed_dict = dict()
        for i, placeholder in enumerate(self.dense_update_placehoders):
            feed_dict[placeholder] = var[i]
        self.sess.run(self.dense_update_ops, feed_dict=feed_dict)

    def _connect_db(self):
        conn = p2.connect(host=self.host, user=self.user, dbname=self.dbname, port=self.port)
        return conn

    def _execute(self, sql):
        conn = self._connect_db()
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        if cursor: cursor.close()
        if conn: conn.close()

    def _fetch_results(self, sql, json=False):
        conn = self._connect_db()
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            if cursor: cursor.close()
            if conn: conn.close()

            if json:
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in results]
            return results
        except Exception as e:
            print(e)
            return None

    def check_table_exists(self, name):
        sql_check_exists = "SELECT EXISTS( SELECT 1 FROM pg_class, pg_namespace WHERE relnamespace = pg_namespace.oid AND relname='{name}') as table_exists;".format(
            **locals())
        return self._fetch_results(sql_check_exists)[0][0]

    def clear(self):
        sql = "select pid, query from pg_stat_activity where datname='{self.sample_tbl}';".format(**locals())
        results = self._fetch_results(sql)
        for row in results:
            pid, query = row
            if not 'pg_stat_activity' in query:
                self._execute("select pg_terminate_backend({pid})".format(**locals()))

    def register_model(self):
        self.model_id = 1
        flag = 0
        cnn_weights_table = 'cnn_weights_table'
        if not self.check_table_exists(cnn_weights_table):
            colnames = ['model_id', 'worker_id', 'weight', 'shape', 'name', 'description']
            coltypes = ['int', 'int', 'bytea', 'Text', 'TEXT', 'TEXT']
            col_defs = ','.join(map(' '.join, zip(colnames, coltypes)))
            sql = "CREATE TABLE cnn_weights_table ({col_defs})".format(**locals())
            self._execute(sql)
            flag = 1
        else:
            record("Table {} exists\n".format(cnn_weights_table))
        if flag == 1:
            variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            variables_value = [self.sess.run(v) for v in variables]
            shapes = [v.shape.as_list() for v in variables]
            master_id = 4
            conn = self._connect_db()
            cursor = conn.cursor()
            for i in range(len(variables)):
                weight_serialized = tf_serialize_nd_weights(variables_value[i])
                shape_i = shapes[i]
                record("dense model weights with {} and shape {} at layer {}\n".format(sys.getsizeof(weight_serialized),shapes[i],i))
                model_id = i
                sql_insert_dense =" INSERT INTO {} VALUES({}, {}, %s, %s )".format(cnn_weights_table, model_id, master_id)
                cursor.execute(sql_insert_dense, (p2.Binary(weight_serialized), str(shape_i)))
            conn.commit()

        fn_weights_table = 'fn_weights_table'
        if not self.check_table_exists(fn_weights_table):
            colnames = ['model_id', 'worker_id', 'weight', 'shape', 'name', 'description']
            coltypes = ['int', 'int', 'bytea', 'Text', 'TEXT', 'TEXT']
            col_defs = ','.join(map(' '.join, zip(colnames, coltypes)))
            sql = "CREATE TABLE fn_weights_table ({col_defs})".format(**locals())
            self._execute(sql)
        else:
            record("Table {} exists\n".format(fn_weights_table))

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
            with tf.variable_scope('conv1_1') as scope:
                kernel = tf.get_variable('weights',
                                         initializer=tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1),
                                         trainable=True)
                conv = tf.nn.conv2d(self.input_tensor, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[64], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv1_1 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)
            # conv1_2
            with tf.variable_scope('conv1_2') as scope:
                kernel = tf.get_variable('weights',
                                         initializer=tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1),
                                         trainable=True)
                conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[64], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv1_2 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)
            # pool1
            pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

            # conv2_1
            with tf.variable_scope('conv2_1') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv2_1 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)
            # conv2_2
            with tf.variable_scope('conv2_2') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv2_2 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # pool2
            pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

            # conv3_1
            with tf.variable_scope('conv3_1') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv3_1 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # conv3_2
            with tf.variable_scope('conv3_2') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv3_2 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # conv3_3
            with tf.variable_scope('conv3_3') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv3_3 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # pool3
            pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

            # conv4_1
            with tf.variable_scope('conv4_1') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv4_1 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # conv4_2
            with tf.variable_scope('conv4_2') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv4_2 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # conv4_3
            with tf.variable_scope('conv4_3') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv4_3 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # pool4
            pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

            # conv5_1
            with tf.variable_scope('conv5_1') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv5_1 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # conv5_2
            with tf.variable_scope('conv5_2') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv5_2 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # conv5_3
            with tf.variable_scope('conv5_3') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv5_3 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # pool5
            pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
            self.shape = int(np.prod(pool5.get_shape()[1:]))
            pool5_flat = tf.reshape(pool5, [-1, self.shape])

            # fc6
            with tf.variable_scope('fc6_par') as scope:
                pfc6w = tf.get_variable('weights', initializer=tf.truncated_normal([self.shape, self.dense_shape],
                                                                                   dtype=tf.float32,
                                                                                   stddev=1e-1))
                pfc6b = tf.get_variable('biases', initializer=tf.constant(1.0, shape=[self.dense_shape], dtype=tf.float32))
                pfc6l = tf.nn.bias_add(tf.matmul(pool5_flat, pfc6w), pfc6b)
                pfc6 = tf.nn.relu(pfc6l)
                tf.add_to_collection('cnn_weights', pfc6w)
                tf.add_to_collection('cnn_weights', pfc6b)
            # fc7
            with tf.variable_scope('fc7_par') as scope:
                pfc7w = tf.get_variable('weights',
                                        initializer=tf.truncated_normal([self.dense_shape, 1024], dtype=tf.float32,
                                                                        stddev=1e-1))
                pfc7b = tf.get_variable('biases', initializer=tf.constant(1.0, shape=[1024], dtype=tf.float32))
                pfc7l = tf.nn.bias_add(tf.matmul(pfc6, pfc7w), pfc7b)
                pfc7 = tf.nn.relu(pfc7l)
                tf.add_to_collection('cnn_weights', pfc7w)
                tf.add_to_collection('cnn_weights', pfc7b)
            # fc8
            with tf.variable_scope('fc8_par') as scope:
                pfc8w = tf.get_variable('weights',
                                        initializer=tf.truncated_normal([1024, self.num_classes], dtype=tf.float32,
                                                                        stddev=1e-1))
                pfc8b = tf.get_variable('biases',
                                        initializer=tf.constant(1.0, shape=[self.num_classes], dtype=tf.float32))
                self.pscore = tf.nn.bias_add(tf.matmul(pfc7, pfc8w), pfc8b)
                tf.add_to_collection('cnn_weights', pfc8w)
                tf.add_to_collection('cnn_weights', pfc8b)
            self.y_label = tf.placeholder(tf.float32, [None, self.num_classes])

            self.ploss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pscore, labels=self.y_label))
            #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,epsilon=1e-8)
            trainable_vars = tf.trainable_variables()
            self.train_op = self.optimize(learning_rate=self.learning_rate,train_layers = trainable_vars)
            #self.train_op = self.optimizer.minimize(self.ploss)
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()
            self.sess = self._init_session()
            
    def init_val(self):
        self.sess.run(self.init)
      
    def cnn_weights_load(self):
        sql = "select weight,shape from cnn_weights_table"
        cnn_res = self._fetch_results(sql)
        cnn_weights = cnn_res[0][0]
        cnn_shapes = cnn_res[0][1]
        cnn_weights = tf_deserialize_as_nd_weights(cnn_weights,cnn_shapes)
        variables_ = list()
        for v in cnn_weights:
            variables_.append(v)
        feed_dict = dict()
        for i, placeholder in enumerate(self.cnn_update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.cnn_update_ops, feed_dict=feed_dict)
    
    def random_split_array(self, m, num_subsets, axis=0):
        idx = np.random.permutation(m.shape[axis])
        m_shuffled = np.take(m, idx, axis=axis)
        subsets = np.split(m_shuffled, num_subsets, axis=axis)
        return subsets

    def fn_partition(self,seg_num):
        fn_par = self.graph.get_collection('fn_par')
        variables_value = [self.sess.run(v) for v in fn_par]

        W1 = variables_value[0]
        W2 = variables_value[1]
        W3 = variables_value[2]
        W1_subsets = self.random_split_array(W1,seg_num,axis=1)
        W2_subsets = self.random_split_array(W2,seg_num,axis=0)
        W3_subsets = self.random_split_array(W3,seg_num,axis=0)
        '''with tf.Session() as session:
            shuffled_indices = tf.random.shuffle(tf.range(tf.shape(W1)[1]))
            W1_shuffled = tf.gather(W1, shuffled_indices)
            W1_subsets = tf.split(W1_shuffled, seg_num, axis=1)
            W1_subsets_val = session.run(W1_subsets)
            #shuffled_indices = tf.random.shuffle(tf.range(tf.shape(W2)[0]))
            #W2_shuffled = tf.gather(W2, shuffled_indices)
            W2_subsets = tf.split(W2, seg_num, axis=0)
            W2_subsets_val = session.run(W2_subsets)
            shuffled_indices = tf.random.shuffle(tf.range(tf.shape(W3)[0]))
            W3_shuffled = tf.gather(W3, shuffled_indices)
            W3_subsets = tf.split(W3_shuffled, seg_num, axis=1)
            W3_subsets_val = session.run(W3_subsets)'''
        
        fn_2 = self.graph.get_collection('fn_2')
        fn_2 = [self.sess.run(v) for v in fn_2]

        shapes = list()
        shapes.append(list(W1_subsets[0].shape))
        shapes.append(list(W2_subsets[0].shape))
        shapes.append(list(W3_subsets[0].shape))
        for v in fn_2:
            shapes.append(list(v.shape))
        
        dense_model_table = 'fn_weights_table'
        for i in range(seg_num):
            weights = list()
            weights.append(W1_subsets[i])
            weights.append(W2_subsets[i])
            weights.append(W3_subsets[i])
            for v in fn_2:
                weights.append(v)
            flattened_weights = [np.asarray(w,dtype=np.float32).tostring() for w in weights]
            weight_serialized = "".join(flattened_weights)
            
            sql = "SELECT model_id FROM fn_weights_table WHERE worker_id = {i}".format(
                **locals())
            result = self._fetch_results(sql)
            if result == []:
                conn = self._connect_db()
                cursor = conn.cursor()
                sql_insert = '''INSERT INTO {dense_model_table} (model_id, worker_id, weight, shape) VALUES ({self.model_id}, {i}, %s, %s)'''.format(
                    **locals())
                cursor.execute(sql_insert, (p2.Binary(weight_serialized),str(shapes)))
                conn.commit()
            else:
                conn = self._connect_db()
                cursor = conn.cursor()
                sql_insert = '''UPDATE {dense_model_table} SET (model_id, weight) = ({self.model_id}, %s) WHERE worker_id = {i}'''.format(
                    **locals())
                cursor.execute(sql_insert, (p2.Binary(weight_serialized),))
                conn.commit()

    def evaluate(self,X,Y):
        feed_dict = {self.input_tensor: X,
                     self.y_label: Y}
        loss,logits = self.sess.run([self.ploss,self.pscore],feed_dict=feed_dict)

        predictions = np.argmax(logits, axis=1)

        Y = np.argmax(Y,axis=1)

        metric = accuracy_score(Y, predictions)
        return loss, metric

    def redistribution(self,seg_num):
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        variables_value = [self.sess.run(v) for v in variables]
        shapes = [v.shape.as_list() for v in variables]
        layer_weights = []
        layer_num = self.master_layer_num
        for j in range(layer_num):
            layer_j = []
            for i in range(seg_num):
                sql_weight = "select weight from cnn_weights_table where worker_id = {i} and model_id = {j}".format(**locals())
                res = self._fetch_results(sql)
                weight = tf_deserialize_as_nd_weights(res[0][0],shapes[j])
                layer_j.append(weight)
            layer_j = np.array(layer_j).mean(axis = 0)
            layer_weights.append(layer_j)
        variables_ = layer_weights
        feed_dict = dict()
        for i, placeholder in enumerate(self.dense_update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.dense_update_ops, feed_dict=feed_dict)

    def update_model_table(self):
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        variables_value = [self.sess.run(v) for v in variables]
        shapes = [v.shape.as_list() for v in variables]
        master_id = 4
        conn = self._connect_db()
        cursor = conn.cursor()
        for i in range(len(variables)):
            weight_serialized = tf_serialize_nd_weights(variables_value)
            record("dense model weights with {} at layer {}\n".format(sys.getsizeof(weight_serialized),i))
            model_id = i
            sql_insert_dense ="UPDATE {} SET (weight) = %s where model_id = {} and worker_id = {}".format(cnn_weights_table, model_id, master_id)
            cursor.execute(sql_insert_dense, (p2.Binary(weight_serialized),))
        conn.commit()

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

class VggNetModel_Worker_new(object):

    def __init__(self, worker_id, num_classes=18, learning_rate=0.01, seg_num=4, dropout_keep_prob=1):
        self.host = '172.17.0.7'
        self.user = 'gpadmin'
        self.dbname = 'gpadmin'
        self.port = 5432
        self.random_seed = 2016
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dense_shape = 1024
        self.dropout_keep_prob = dropout_keep_prob
        self.worker_id = worker_id
        self.model_id = 1
        self.version = self.model_id
        self._init_graph()
        self.init_val()
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.layer_num = len(variables)
        #cnn_weights = self.graph.get_collection('cnn_weights')
        self.dense_update_placehoders = list()
        self.dense_update_ops = list()
        self.cnn_update_placehoders = list()
        self.cnn_update_ops = list()
        with self.graph.as_default():
            for i, variable_ in enumerate(variables):
                dense_placehoder_temp = tf.placeholder(variable_.dtype)
                self.dense_update_placehoders.append(dense_placehoder_temp)
                self.dense_update_ops.append(tf.assign(variable_, dense_placehoder_temp, validate_shape=False))
            '''for i, variable_ in enumerate(cnn_weights):
                cnn_placehoder_temp = tf.placeholder(variable_.dtype)
                self.cnn_update_placehoders.append(cnn_placehoder_temp)
                self.cnn_update_ops.append(tf.assign(variable_, cnn_placehoder_temp, validate_shape=False))'''
        #self.cnn_weights_load()

    def optimize(self, learning_rate, train_layers=[]):
        var_list = [v for v in tf.trainable_variables()]
        return tf.train.AdamOptimizer(learning_rate).minimize(self.ploss, var_list=var_list)
        #return tf.train.GradientDescentOptimizer(learning_rate).minimize(self.ploss, var_list=var_list)

    def load_original_weights(self, skip_layers=[]):
        weights = np.load('vgg16_weights.npz')
        keys = sorted(weights.keys())
        var = list()
        for key in keys:
            var.append(weights[key])
        feed_dict = dict()
        for i, placeholder in enumerate(self.dense_update_placehoders):
            feed_dict[placeholder] = var[i]
        self.sess.run(self.dense_update_ops, feed_dict=feed_dict)

    def _connect_db(self):
        conn = p2.connect(host=self.host, user=self.user, dbname=self.dbname, port=self.port)
        return conn

    def _execute(self, sql):
        conn = self._connect_db()
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        if cursor: cursor.close()
        if conn: conn.close()

    def _fetch_results(self, sql, json=False):
        conn = self._connect_db()
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            if cursor: cursor.close()
            if conn: conn.close()

            if json:
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in results]
            return results
        except Exception as e:
            print(e)
            return None

    def check_table_exists(self, name):
        sql_check_exists = "SELECT EXISTS( SELECT 1 FROM pg_class, pg_namespace WHERE relnamespace = pg_namespace.oid AND relname='{name}') as table_exists;".format(
            **locals())
        return self._fetch_results(sql_check_exists)[0][0]

    def cnn_weights_load(self):
        sql = "select weight,shape from cnn_weights_table"
        cnn_res = self._fetch_results(sql)
        cnn_weights = cnn_res[0][0]
        cnn_shapes = cnn_res[0][1]
        cnn_weights = tf_deserialize_as_nd_weights(cnn_weights,cnn_shapes)
        variables_ = list()
        for v in cnn_weights:
            variables_.append(v)
        feed_dict = dict()
        for i, placeholder in enumerate(self.cnn_update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.cnn_update_ops, feed_dict=feed_dict)

    def weights_load(self):
        variables_ = list()
        for i in range(self.layer_num):
            sql = "select weight,shape from cnn_weights_table where worker_id = 4 and model_id = {}".format(i)
            dense_res = self._fetch_results(sql)
            #record("dense model weights shape {} at layer {}\n".format(dense_res[0][1],i))
            dense_weights = tf_deserialize_as_nd_weights(dense_res[0][0], dense_res[0][1])
            variables_.append(dense_weights)
        feed_dict = dict()
        for i, placeholder in enumerate(self.dense_update_placehoders):
            feed_dict[placeholder] = variables_[i]
        record("Worker {} get dense\n".format(self.worker_id))
        self.sess.run(self.dense_update_ops, feed_dict=feed_dict)

    def dense_load(self):
        sql = "select weight,shape from fn_weights_table where worker_id = {self.worker_id}".format(**locals())
        dense_res = self._fetch_results(sql)
        dense_weights = tf_deserialize_as_nd_weights(dense_res[0][0], dense_res[0][1])
        variables_ = list()
        for v in dense_weights:
            variables_.append(v)
        feed_dict = dict()
        for i, placeholder in enumerate(self.dense_update_placehoders):
            feed_dict[placeholder] = variables_[i]
        record("Worker {} get dense\n".format(self.worker_id))
        self.sess.run(self.dense_update_ops, feed_dict=feed_dict)


    def clear(self):
        sql = "select pid, query from pg_stat_activity where datname='{self.sample_tbl}';".format(**locals())
        results = self._fetch_results(sql)
        for row in results:
            pid, query = row
            if not 'pg_stat_activity' in query:
                self._execute("select pg_terminate_backend({pid})".format(**locals()))

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
            with tf.variable_scope('conv1_1') as scope:
                kernel = tf.get_variable('weights',
                                         initializer=tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1),
                                         trainable=True)
                conv = tf.nn.conv2d(self.input_tensor, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[64], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv1_1 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)
            # conv1_2
            with tf.variable_scope('conv1_2') as scope:
                kernel = tf.get_variable('weights',
                                         initializer=tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1),
                                         trainable=True)
                conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[64], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv1_2 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)
            # pool1
            pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

            # conv2_1
            with tf.variable_scope('conv2_1') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv2_1 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)
            # conv2_2
            with tf.variable_scope('conv2_2') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv2_2 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # pool2
            pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

            # conv3_1
            with tf.variable_scope('conv3_1') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv3_1 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # conv3_2
            with tf.variable_scope('conv3_2') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv3_2 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # conv3_3
            with tf.variable_scope('conv3_3') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv3_3 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # pool3
            pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

            # conv4_1
            with tf.variable_scope('conv4_1') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv4_1 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # conv4_2
            with tf.variable_scope('conv4_2') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv4_2 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # conv4_3
            with tf.variable_scope('conv4_3') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv4_3 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # pool4
            pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

            # conv5_1
            with tf.variable_scope('conv5_1') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv5_1 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # conv5_2
            with tf.variable_scope('conv5_2') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv5_2 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # conv5_3
            with tf.variable_scope('conv5_3') as scope:
                kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                                    stddev=1e-1), trainable=True)
                conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True)
                out = tf.nn.bias_add(conv, biases)
                conv5_3 = tf.nn.relu(out, name=scope.name)
                tf.add_to_collection('cnn_weights', kernel)
                tf.add_to_collection('cnn_weights', biases)

            # pool5
            pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
            self.shape = int(np.prod(pool5.get_shape()[1:]))
            pool5_flat = tf.reshape(pool5, [-1, self.shape])

            # fc6
            with tf.variable_scope('fc6_par') as scope:
                pfc6w = tf.get_variable('weights', initializer=tf.truncated_normal([self.shape, self.dense_shape],
                                                                                   dtype=tf.float32,
                                                                                   stddev=1e-1))
                pfc6b = tf.get_variable('biases', initializer=tf.constant(1.0, shape=[self.dense_shape], dtype=tf.float32))
                pfc6l = tf.nn.bias_add(tf.matmul(pool5_flat, pfc6w), pfc6b)
                pfc6 = tf.nn.relu(pfc6l)

            # fc7
            with tf.variable_scope('fc7_par') as scope:
                pfc7w = tf.get_variable('weights',
                                        initializer=tf.truncated_normal([self.dense_shape, 1024], dtype=tf.float32,
                                                                        stddev=1e-1))
                pfc7b = tf.get_variable('biases', initializer=tf.constant(1.0, shape=[1024], dtype=tf.float32))
                pfc7l = tf.nn.bias_add(tf.matmul(pfc6, pfc7w), pfc7b)
                pfc7 = tf.nn.relu(pfc7l)

            # fc8
            with tf.variable_scope('fc8_par') as scope:
                pfc8w = tf.get_variable('weights',
                                        initializer=tf.truncated_normal([1024, self.num_classes], dtype=tf.float32,
                                                                        stddev=1e-1))
                pfc8b = tf.get_variable('biases',
                                        initializer=tf.constant(1.0, shape=[self.num_classes], dtype=tf.float32))
                self.pscore = tf.nn.bias_add(tf.matmul(pfc7, pfc8w), pfc8b)
            self.y_label = tf.placeholder(tf.float32, [None, self.num_classes])

            self.ploss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pscore, labels=self.y_label))
            
            #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,epsilon=1e-8)
            #self.train_op = self.optimizer.minimize(self.ploss)
            trainable_vars = tf.trainable_variables()
            self.train_op = self.optimize(learning_rate=self.learning_rate,train_layers = trainable_vars)
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()
            self.sess = self._init_session()
            
    def init_val(self):
        self.sess.run(self.init)
    
    def train(self, X, y):
        #variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #variables_value = [self.sess.run(v) for v in variables]
        #record("train_before {}\n".format(variables_value[-1]))
        feed_dict = {self.input_tensor: X,
                     self.y_label: y}
        train_op,loss = self.sess.run([self.train_op,self.ploss],feed_dict=feed_dict)
        
        return loss
    
    def evaluate(self,X,Y):
        feed_dict = {self.input_tensor: X,
                     self.y_label: Y}
        loss,logits = self.sess.run([self.ploss,self.pscore],feed_dict=feed_dict)

        predictions = np.argmax(logits, axis=1)

        Y = np.argmax(Y,axis=1)

        metric = accuracy_score(Y, predictions)
        return loss, metric
    
    def push_dense_weights(self):
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        variables_value = [self.sess.run(v) for v in variables]
        for layer_id in range(self.layer_num):
            flattened_weights = np.float32(variables_value[layer_id]).tostring() 
            flattened_weights = "".join(flattened_weights)
            dense_model_table = 'cnn_weights_table'
            sql = "SELECT shape FROM {dense_model_table} WHERE worker_id = {self.worker_id} and model_id = {layer_id}".format(
                **locals())
            result = self._fetch_results(sql)
            if result == []:
                conn = self._connect_db()
                cursor = conn.cursor()
                sql_insert = '''INSERT INTO {dense_model_table} (model_id, worker_id, weight) VALUES ({layer_id}, {self.worker_id}, %s)'''.format(
                    **locals())
                cursor.execute(sql_insert, (p2.Binary(flattened_weights),))
                conn.commit()
            else:
                conn = self._connect_db()
                cursor = conn.cursor()
                sql_insert = '''UPDATE {dense_model_table} SET (weight) = (%s) WHERE worker_id = {self.worker_id} and model_id = {layer_id}'''.format(
                    **locals())
                cursor.execute(sql_insert, (p2.Binary(flattened_weights),))
                conn.commit()
        record("Worker {} push dense \n".format(self.worker_id))

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)
