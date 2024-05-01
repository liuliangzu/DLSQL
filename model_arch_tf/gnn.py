import numpy as np
import pandas as pd
import tensorflow as tf
import psycopg2 as p2
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.python.framework import sparse_tensor
import ast

class GCN_model(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, num_classes, field_size, embedding_size, verbose=False, random_seed=2016,
                 gcn_layers=2, learning_rate=0.001, *args, **kwargs):
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.random_seed = random_seed
        self.gcn_layers = gcn_layers
        self.field_size = field_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.verbose = verbose
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.feature_index = tf.placeholder(tf.int32, shape=[None, self.field_size], name="feat_index")  # None * F
            self.label = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="label")  # None * 1
            self.weights = self._initialize_weights()
            self.adj_matrix = tf.placeholder(tf.float32, shape=[None, None], name='adj_matrix')  # None * None

            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"],
                                                     self.feature_index)  # None * F * K
            self.embeddings_bias = tf.nn.embedding_lookup(self.weights["feature_bias"], self.feature_index)
            self.embeddings = tf.reshape(self.embeddings, [-1, self.field_size * self.embedding_size])
            kernel_support = tf.matmul(self.embeddings, self.weights["kernel_0"])
            kernel_output = tf.matmul(self.adj_matrix, kernel_support)
            kernel_output = tf.nn.relu(kernel_output)
            kernel_support = tf.matmul(kernel_output, self.weights["kernel_1"])
            kernel_output = tf.matmul(self.adj_matrix, kernel_support)
            kernel_output = tf.nn.relu(kernel_output)
            output_without_bias = tf.matmul(kernel_output, self.weights["layer_weights"])
            self.out = tf.nn.bias_add(output_without_bias, self.weights["layer_bias"])
            self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def _initialize_weights(self):
        weights = dict()
        # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # feature_size * K
        weights["feature_bias"] = tf.Variable(
            tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name="feature_bias")  # feature_size * 1

        # input_size = self.field_size * self.embedding_size
        weights["kernel_0"] = tf.Variable(
            tf.random_uniform([self.field_size * self.embedding_size, 128], 0.0, 1.0), name="gcn_kernel")
        weights["bias_0"] = tf.Variable(
            tf.random_uniform([1, 32], 0.0, 1.0), name="gcn_bias")
        weights["kernel_1"] = tf.Variable(
            tf.random_uniform([128, 32], 0.0, 1.0), name="gcn_kernel")
        weights["bias_1"] = tf.Variable(
            tf.random_uniform([1, 32], 0.0, 1.0), name="gcn_bias")

        weights["layer_weights"] = tf.Variable(
            tf.random_uniform([32, self.num_classes], 0.0, 1.0), name="layer_weights")
        weights["layer_bias"] = tf.Variable(
            tf.random_uniform([self.num_classes], 0.0, 1.0), name="layer_bias")
        return weights

    def gradients_compute(self, X, Y, adj):

        feed_dict = {
            self.feature_index: X,
            self.label: Y,
            self.adj_matrix:adj
        }
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)

        print("gnn loss:{}".format(loss))

class GCN_model_DA(GCN_model):
    def __init__(self, **kwargs):
        self.user = 'gpadmin'
        self.host = '172.17.31.87'
        self.dbname = 'gpadmin'
        self.sample_tbl = 'driving'
        self.port = 5432
        self.Xi_train, self.Xv_train, self.y_train = list(), list(), list()        
        GCN_model.__init__(self, **kwargs)
        record("Initialized model\n")

    def _connect_db(self):
        conn = p2.connect(host=self.host, user=self.user, dbname=self.dbname, port=self.port)
        return conn

    def _connect_seg_db(self, seg_port, seg_ip):
        conn = p2.connect(host=seg_ip, user=self.user, dbname=self.dbname, port=seg_port, options='-c gp_session_role=utility')
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
            return  None

    def _fetch_results_onseg(self, sql, json=False):
        conn = self._connect_seg_db(6000,'127.0.0.1')
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
            return  None

    def check_table_exists(self, name):
        sql_check_exists = "SELECT EXISTS( SELECT 1 FROM pg_class, pg_namespace WHERE relnamespace = pg_namespace.oid AND relname='{name}') as table_exists;".format(**locals())
        return self._fetch_results(sql_check_exists)[0][0]

    def clear(self):
        sql = "select pid, query from pg_stat_activity where datname='{self.sample_tbl}';".format(**locals())
        results = self._fetch_results(sql)
        for row in results:
            pid, query = row
            if not 'pg_stat_activity' in query:
                self._execute("select pg_terminate_backend({pid})".format(**locals()))
                
class GCN_model_Master(GCN_model_DA):
    def __init__(self, **kwargs):
        GCN_model_DA.__init__(self, **kwargs)
        self.dict_mapping = None
        self.hot_id = None
        self.embedding_cache_dict = dict()
        self.embedding_bias_cache_dict = dict()
        self.embedding_dict = set()
        self.clear()
        self.register_model()
        self.version = dict()
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.update_placehoders = list()
        self.update_ops = list()
        self.gradient_placehoders = list()
        gradientL = list()
        self.dense_update_placehoders = list()
        self.dense_update_ops = list()
        embed_model_table, dense_model_table, embed_gradient_table, dense_gradient_table =Schema.Embed_Model_Table, Schema.Dense_Model_Table, Schema.Embed_GRADIENT_TABLE, Schema.Dense_GRADIENT_TABLE
        self.model_id = self._fetch_results("SELECT max(model_id) from {embed_model_table}".format(**locals()))[0][0]
        with self.graph.as_default():
            for i, variable_ in enumerate(variables):
                if i < 2:
                    placehoder_temp = tf.placeholder(variable_.dtype)
                    self.update_placehoders.append(placehoder_temp)
                    self.update_ops.append(tf.assign(variable_, placehoder_temp, validate_shape=False))
                    placehoder_value = tf.placeholder(variable_.dtype)
                    placehoder_indices = tf.placeholder('int64')
                    self.gradient_placehoders.append(placehoder_value)
                    self.gradient_placehoders.append(placehoder_indices)
                    gradientL.append(tf.IndexedSlices(placehoder_value,placehoder_indices))
                else:
                    dense_placehoder_temp = tf.placeholder(variable_.dtype)
                    self.dense_update_placehoders.append(dense_placehoder_temp)
                    self.dense_update_ops.append(tf.assign(variable_, dense_placehoder_temp, validate_shape=False))

            self.embed_apply_grad_op = self.optimizer.apply_gradients(zip(gradientL, variables[0:2]))

        record("Init GCN_model_Master in DB\n")
    def register_model(self, name='', description=''):
        embed_model_table, dense_model_table, embed_gradient_table, dense_gradient_table =Schema.Embed_Model_Table, Schema.Dense_Model_Table, Schema.Embed_GRADIENT_TABLE, Schema.Dense_GRADIENT_TABLE
        if not self.check_table_exists(embed_model_table):
            colnames = ['id', 'embedding_weight', 'shape', 'embedding_bias', 'model_id', 'description']
            coltypes = ['int', 'bytea', 'Text', 'bytea', 'int', 'TEXT']
            col_defs = ','.join(map(' '.join, zip(colnames, coltypes)))
            sql = "CREATE TABLE {embed_model_table} ({col_defs})".format(**locals())
            self._execute(sql)
        else:
            record("Table {} exists\n".format(embed_model_table))

        if not self.check_table_exists(dense_model_table):
            colnames = ['model_id', 'worker_id', 'weight', 'shape', 'name', 'description']
            coltypes = ['int', 'int', 'bytea', 'Text', 'TEXT', 'TEXT']
            col_defs = ','.join(map(' '.join, zip(colnames, coltypes)))
            sql = "CREATE TABLE {dense_model_table} ({col_defs})".format(**locals())
            self._execute(sql)
        else:
            record("Table {} exists\n".format(dense_model_table))

        if not self.check_table_exists(embed_gradient_table):
            colnames = ['model_id', 'worker_id', 'gradient', 'shape', 'version', 'model_version', 'auxiliaries']
            coltypes = ['int', 'int', 'bytea', 'Text', 'int', 'int', 'Text']
            col_defs = ','.join(map(' '.join, zip(colnames, coltypes)))
            sql = "CREATE TABLE {embed_gradient_table} ({col_defs})".format(**locals())
            self._execute(sql)
        else:
            record("Table {} exists\n".format(embed_gradient_table))

        if not self.check_table_exists(dense_gradient_table):
            colnames = ['model_id', 'worker_id', 'gradient', 'shape', 'version', 'model_version','auxiliaries']
            coltypes = ['int', 'int', 'bytea', 'Text', 'int', 'int', 'Text']
            col_defs = ','.join(map(' '.join, zip(colnames, coltypes)))
            sql = "CREATE TABLE {dense_gradient_table} ({col_defs})".format(**locals())
            self._execute(sql)
        else:
            record("Table {} exists\n".format(dense_gradient_table))

        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[2:]
        variables_value = [self.sess.run(v) for v in variables]
        shapes = [v.shape.as_list() for v in variables]
        weight_serialized = tf_serialize_nd_weights(variables_value)
        start_id = 1
        master_id = 6
        sql_insert_dense = '''INSERT INTO {} VALUES({}, {}, %s, %s )'''.format(dense_model_table, start_id, master_id)
        conn = self._connect_db()
        cursor = conn.cursor()
        cursor.execute(sql_insert_dense, (p2.Binary(weight_serialized), str(shapes)))
        conn.commit()
        self.model_id = self._fetch_results("SELECT max(model_id) from {dense_model_table}".format(**locals()))[0][0]
        record("Register model {} in DB\n".format(self.model_id))
    
    def dense_weights_transformer(self):
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[2:]
        variables_value = [self.sess.run(v) for v in variables]
        shapes = [v.shape.as_list() for v in variables]
        weight_serialized = tf_serialize_nd_weights(variables_value)
        return weight_serialized

    def save_embedding(self, weights, embed_id):
        conn = self._connect_db()
        cursor = conn.cursor()
        create_tmp_table_Sql = '''CREATE TEMP TABLE temp_embed (
        id INTEGER PRIMARY KEY,
        embedding_weight BYTEA,
        embedding_bias BYTEA
    );
    '''
        cursor.execute(create_tmp_table_Sql)
        update_data = list()
        for i, j, k in zip(weights[0], weights[1], embed_id):
            tmp = []
            tmp.append(k)
            tmp.append(str(p2.Binary(np.float32(i).tostring())))
            tmp.append(str(p2.Binary(np.float32(j).tostring())))
            update_data.append(tmp)
        values = []
        for data in update_data:
            values.append("({}, {}, {})".format(data[0], data[1], data[2]))
        print(values[0])
        sql = "INSERT INTO temp_embed(id, embedding_weight, embedding_bias) VALUES " + ",".join(values)
        cursor.execute(sql)
        update_Sql = '''UPDATE embed_model
    SET embedding_weight = temp_embed.embedding_weight,
        embedding_bias = temp_embed.embedding_bias
    FROM temp_embed
    WHERE embed_model.id = temp_embed.id;
    '''
        cursor.execute(update_Sql)
        conn.commit()
    
    def save_embedding_with_cache(self):
        conn = self._connect_db()
        cursor = conn.cursor()
        create_tmp_table_Sql = '''CREATE TEMP TABLE temp_embed (
        id INTEGER PRIMARY KEY,
        embedding_weight BYTEA,
        embedding_bias BYTEA
    );
    '''
        cursor.execute(create_tmp_table_Sql)
        update_data = list()
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0:2]
        weights = [self.sess.run(v) for v in variables]
        for key in list(self.embedding_dict):
            embedding = weights[0][key]
            embedding_bias = weights[1][key]
            tmp = []
            tmp.append(key)
            tmp.append(str(p2.Binary(np.float32(embedding).tostring())))
            tmp.append(str(p2.Binary(np.float32(embedding_bias).tostring())))
            update_data.append(tmp)
        del weights,variables
        for i in range(len(list(self.embedding_dict))/100000 + 1):
            values = []
            start = i * 100000
            end = start + 100000
            if end > len(list(self.embedding_dict)):
                end = len(list(self.embedding_dict))
            for data in update_data[start:end]:
                values.append("({}, {}, {})".format(data[0], data[1], data[2]))
            sql = "INSERT INTO temp_embed(id, embedding_weight, embedding_bias) VALUES " + ",".join(values)
            cursor.execute(sql)
        update_Sql = '''UPDATE embed_model
    SET embedding_weight = temp_embed.embedding_weight,
        embedding_bias = temp_embed.embedding_bias
    FROM temp_embed
    WHERE embed_model.id = temp_embed.id;
    '''
        cursor.execute(update_Sql)
        conn.commit()
        pass
    
    def save_dense_weight(self, weights_serialized):
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        weights_updated = [self.sess.run(v) for v in variables]
        shapes = [v.shape for v in weights_updated[2:]]

        dense_weights = tf_deserialize_as_nd_weights(weights_serialized,shapes)
        variables_ = list()
        for v in dense_weights:
            variables_.append(v)
        feed_dict = dict()
        for i, placeholder in enumerate(self.dense_update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.dense_update_ops, feed_dict=feed_dict)

        sql_insert = '''UPDATE {} SET (model_id, weight, shape) = ({}, %s, %s) where worker_id = 6'''.format(Schema.Dense_Model_Table, self.model_id)
        conn = self._connect_db()
        cursor = conn.cursor()
        cursor.execute(sql_insert, (p2.Binary(weights_serialized), str(shapes)))
        record("[Master] Save Dense at [{}]\n".format(time.time()))
        conn.commit()

    def update(self, grads, embed_id):
        t1 = time.time()
        dense_result = self._fetch_results("SELECT weight, shape FROM {} WHERE model_id ={}".format(Schema.Dense_Model_Table, self.model_id))
        shapes_fetch = eval(dense_result[0][1])
        dense_weights = tf_deserialize_as_nd_weights(dense_result[0][0], shapes_fetch)
        t2 = time.time()
        record("[Master] Pull dense [{} s]".format(round(t2-t1, 2)))
        embedding_result = self._fetch_results("SELECT embedding_weight, embedding_bias, id FROM {} WHERE id in {} and model_id={}".format(Schema.Embed_Model_Table, tuple(embed_id), self.model_id))
        emb_id_mapping = dict()
        embedding, embedding_bias = list(), list()
        embed_id_ = list()
        for i, row in enumerate(embedding_result):
            embedding.append(tf_deserialize_embedding(row[0]))
            embedding_bias.append(tf_deserialize_embedding(row[1]))
            emb_id_mapping[row[2]] = i
            embed_id_.append(row[2])
        t3 = time.time()
        record("[Master] Pull embedding [{} s]".format(round(t3-t2, 2)))
        variables_ = list()
        variables_.append(np.array(embedding))
        variables_.append(np.array(embedding_bias))
        for v in dense_weights:
            variables_.append(v)

        feed_dict = dict()
        for i, placeholder in enumerate(self.update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.update_ops, feed_dict=feed_dict)

        feed_dict = dict()
        placeholder_count = 0
        for i, grad_ in enumerate(grads):
            if isinstance(grad_, IndexedSlicesValue):
                indices = np.vectorize(emb_id_mapping.get)(grads[i].indices.astype('int64'))
                feed_dict[self.gradient_placehoders[placeholder_count]] = grads[i].values
                placeholder_count = placeholder_count + 1
                feed_dict[self.gradient_placehoders[placeholder_count]] = indices
                placeholder_count = placeholder_count + 1
            else:
                feed_dict[self.gradient_placehoders[placeholder_count]] = grads[i]
                placeholder_count = placeholder_count + 1

        self.sess.run(self.apply_grad_op, feed_dict=feed_dict)

        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        weights_updated = [self.sess.run(v) for v in variables]
        t4 = time.time()
        record("[Master] Update weight in master [{} s]".format(round(t4-t3, 2)))
        self.save_embedding(weights_updated[0:2], embed_id_)
        t5 = time.time()
        record("[Master] Save embedding [{} s]".format(round(t5-t4, 2)))
        self.save_dense_weight(weights_updated[2:])
        t6 = time.time()
        record("[Master] Save dense [{} s]".format(round(t6-t5, 2)))
        return weights_updated
    
    def pull_embedding_grads(self, worker_id, model_version):
        embed_gradient_table = Schema.Embed_GRADIENT_TABLE
        record("[Master] [Worker{}] Start get gradients Version {}\n ".format(worker_id, model_version))
        query = '''SELECT gradient, shape, auxiliaries FROM {embed_gradient_table} WHERE model_id={model_version} AND worker_id={worker_id}'''.format(
            **locals())
        results = self._fetch_results(query)
        embed_gradient_serialized, shape, auxiliaries = results[0]
        embed_gradients = tf_deserialize_gradient(embed_gradient_serialized, eval(shape), eval(auxiliaries))
        record("[Master] [Worker{}] Receive embedding gradients with version {}\n".format(worker_id, model_version))
        grads = list()
        embed_id = list()
        for e in embed_gradients:
            grads.append(e)
            embed_id.append(e.indices.astype('int64'))
        embed_id_unique = np.unique(np.array(embed_id)).tolist()
        return  grads, embed_id_unique
        
    def apply_embed_grads_per_worker(self, worker_id, model_version):
        t1 = time.time()
        grads, embed_id_unique = self.pull_embedding_grads(worker_id, model_version)
        t2 = time.time()
        record("[Mater] Pull gradient  [{} s]\n".format(round(t2 - t1,2)))
        self.update_embedding_and_save(worker_id, grads, embed_id_unique)
        t3 = time.time()
        record("[Mater] Update weight [{} s]\n".format(round(t3 - t2,2)))
        record("[Master] [Worker{}] Save embed model with version {}\n".format(worker_id, model_version))

class GCN_model_Worker(GCN_model_DA):
    def __init__(self, worker_id, **kwargs):
        GCN_model_DA.__init__(self, **kwargs)
        self.update_time = None
        self.prune_id = None
        self.dict_mapping = None
        self.hot_id = embedding_cache(cache_capcity=100000,update_staleness=10)
        self.worker_id = worker_id
        self.model_id = self._fetch_results("SELECT max(model_id) from dense_model where worker_id = 6".format(**locals()))[0][0]
        self.version = self.model_id
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.embed_update_placehoders = list()
        self.embed_update_ops = list()
        self.dense_update_placehoders = list()
        self.dense_update_ops = list()
        self.gradient_placehoders = list()
        gradientL = list()
        with self.graph.as_default():
            for variable_ in variables[0:2]:
                placehoder_temp = tf.placeholder(variable_.dtype, [None for i in  variable_.shape])
                self.embed_update_placehoders.append(placehoder_temp)
                self.embed_update_ops.append(tf.assign(variable_, placehoder_temp, validate_shape=False))

            for variable_ in variables[2:]:
                placehoder_temp = tf.placeholder(variable_.dtype, [None for i in  variable_.shape])
                self.dense_update_placehoders.append(placehoder_temp)
                self.dense_update_ops.append(tf.assign(variable_, placehoder_temp, validate_shape=False))
                self.gradient_placehoders.append(placehoder_temp)
                gradientL.append(placehoder_temp)

            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.update_all_op = self.optimizer.apply_gradients(grads_and_vars)
            self.grad_op = [x[0] for x in grads_and_vars]
            self.dense_apply_grad_op = self.optimizer.apply_gradients(zip(gradientL, variables[2:]))

        self.emb_thrhold = 100000
        self.init_weight = True
        self.pull_dense_weights()
        self.get_hot_id()
        record("[seg {}] Init DeepFM_worker in DB with model version {}\n".format(worker_id,self.model_id))
        
    def pull_embedding_weights(self, embed_id_unique):
        embedding = list()
        embedding_bias = list()
        record("model pull embedding weights with len {}\n".format(len(embed_id_unique)))
        embed_result = self._fetch_results("SELECT embedding_weight, embedding_bias, id FROM embed_model WHERE id in {}".format(tuple(embed_id_unique)))
        emb_id_mapping = dict()
        record("model pull embedding weights with len {}\n".format(len(embed_id_unique)))
        for i, row in enumerate(embed_result):
            embedding.append(tf_deserialize_embedding(row[0]))
            embedding_bias.append(tf_deserialize_embedding(row[1]))
            emb_id_mapping[row[2]] = i
        variables_ = list()
        variables_.append(np.array(embedding))
        variables_.append(np.array(embedding_bias))

        feed_dict = dict()
        for i, placeholder in enumerate(self.embed_update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.embed_update_ops, feed_dict=feed_dict)

        return emb_id_mapping
    
    def get_hot_id(self):
        hot_id_sql = "Select key from hot_key order by times desc"
        res = self._fetch_results(hot_id_sql)
        hot_id = list()
        for data in res:
            hot_id.append(data[0])
        self.hot_id.init_hot_cache(hot_id)

    def hot_key_init(self):
        hot_id = self.hot_id
        hot_id = np.array(hot_id,dtype=int)
        embedding_pulling_sql = "SELECT embedding_weight, embedding_bias FROM {} WHERE id in {}".format(Schema.Embed_Model_Table, tuple(hot_id))
        res = self._fetch_results(embedding_pulling_sql)
        self.update_time = 0
        embedding = list()
        embedding_bias = list()
        for i, row in enumerate(res):
            embedding.append(tf_deserialize_embedding(row[0]))
            embedding_bias.append(tf_deserialize_embedding(row[1]))
        variables_ = list()
        variables_.append(np.array(embedding))
        variables_.append(np.array(embedding_bias))
        feed_dict = dict()
        for i, placeholder in enumerate(self.embed_update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.embed_update_ops, feed_dict=feed_dict)

    def key_time_check(self, hash_time):
        last_time = self.update_time
        if last_time > hash_time:
            self.hot_key_update()
            self.update_time = 0

    def hot_key_update(self, new_id):
        pulling_id = list(set(new_id) - set(self.hot_id))
        pulling_id = np.array(pulling_id,dtype=int)
        embedding_pulling_sql = "SELECT embedding_weight, embedding_bias, id FROM {} WHERE id in {}".format(Schema.Embed_Model_Table, tuple(pulling_id))
        res = self._fetch_results(embedding_pulling_sql)
        record("{} , {} \n".format(len(pulling_id),len(res)))
        emb_id_mapping = dict()
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[:2]
        weights_updated = [self.sess.run(v) for v in variables]
        embedding = weights_updated[0][0:len(self.hot_id)].tolist()
        embedding_bias = weights_updated[1][0:len(self.hot_id)].tolist()
        for i, row in enumerate(res):
            embedding.append(tf_deserialize_embedding(row[0]))
            embedding_bias.append(tf_deserialize_embedding(row[1]))
            emb_id_mapping[row[2]] = i + len(self.hot_id)
        for i in range(len(self.hot_id)):
            emb_id_mapping[self.hot_id[i]] = i
        variables_ = list()
        variables_.append(np.array(embedding))
        variables_.append(np.array(embedding_bias))
        
        feed_dict = dict()
        for i, placeholder in enumerate(self.embed_update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.embed_update_ops, feed_dict=feed_dict)
        self.dict_mapping = emb_id_mapping
        return emb_id_mapping
        
    def pull_new_key(self, new_id):
        hot_id_list = self.hot_id.hot_list
        pulling_id = list(set(new_id) - set(hot_id_list))
        pulling_id = np.array(pulling_id,dtype=int)
        self.update_time = self.update_time + 1
        embedding_pulling_sql = "SELECT embedding_weight, embedding_bias, id FROM embed_model WHERE id in {}".format(tuple(pulling_id))
        res = self._fetch_results(embedding_pulling_sql)
        emb_id_mapping = dict()
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0:2]
        weights_updated = [self.sess.run(v) for v in variables]
        embedding = list()
        embedding_bias = list()
        for i in range(len(hot_id_list)):
            emb_id_mapping[hot_id_list[i]] = i
            embedding.append(weights_updated[0][i])
            embedding_bias.append(weights_updated[1][i])
        for i, row in enumerate(res):
            embedding.append(tf_deserialize_embedding(row[0]))
            embedding_bias.append(tf_deserialize_embedding(row[1]))
            emb_id_mapping[row[2]] = i + len(hot_id_list)
        variables_ = list()
        variables_.append(np.array(embedding))
        variables_.append(np.array(embedding_bias))
        self.hot_id.put_id(pulling_id=list(pulling_id))
        feed_dict = dict()
        for i, placeholder in enumerate(self.embed_update_placehoders):
            feed_dict[placeholder] = variables_[i]
        self.sess.run(self.embed_update_ops, feed_dict=feed_dict)
        self.dict_mapping = emb_id_mapping
        return emb_id_mapping
    
    def push_embedding_grads_dps(self, batch_id, embedding_grads):
        t1 = time.time()
        embed_grads = embedding_grads
        embed_gradient_table = Schema.Embed_GRADIENT_TABLE
        grads_serialized, shapes, auxiliaries = tf_serialize_gradient(embed_grads)
        sql = "SELECT model_id FROM {embed_gradient_table} WHERE worker_id = {self.worker_id} order by model_id".format(
            **locals())
        result = self._fetch_results(sql)
        target_id = self.model_id + batch_id
        if result == [] or len(result) < 600:
            conn = self._connect_db()
            cursor = conn.cursor()
            sql_insert = '''INSERT INTO {embed_gradient_table} (model_id, worker_id, gradient, shape, version, auxiliaries) VALUES ({target_id}, {self.worker_id}, %s , %s, {self.version}, %s)'''.format(**locals())
            gs = p2.Binary(grads_serialized)
            cursor.execute(sql_insert, (gs, str(shapes), str(auxiliaries)))
            conn.commit()

        else:
            update_id = result[0][0]
            sql_update = '''UPDATE {embed_gradient_table} SET (model_id, gradient, shape, version, auxiliaries) = ({target_id}, %s, %s, {self.version} ,%s) WHERE worker_id={self.worker_id} and model_id = {update_id} '''.format(
                **locals())
            conn = self._connect_db()
            cursor = conn.cursor()
            cursor.execute(sql_update, (p2.Binary(grads_serialized), str(shapes), str(auxiliaries)))
            conn.commit()
        record("[Worker{}] Push embed_gradients with version {} last {}\n".format(self.worker_id, target_id, time.time()-t1))
        
    def pull_dense_weights(self):
        check_dense = 0
        while check_dense==0:
            dense_version = self._fetch_results("SELECT model_id FROM dense_model where worker_id = 6")
            if dense_version[0][0] == self.model_id:
                check_dense = 1
        variables_ = list()
        dense_result = self._fetch_results("SELECT weight, shape, model_id FROM dense_model where worker_id = 6")
        record("[seg] {} pull_dense, worked model_id :{}, dense id : {}\n".format(self.worker_id,self.model_id,dense_result[0][2]))
        shapes_fetch = eval(dense_result[0][1])
        dense_weights = tf_deserialize_as_nd_weights(dense_result[0][0], shapes_fetch)
        for v in dense_weights:
            variables_.append(v)
        feed_dict = dict()
        for i, placeholder in enumerate(self.dense_update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.dense_update_ops, feed_dict=feed_dict)
        
    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    def gradients_compute(self, Xi, Xv, Y, return_grads):
        for i in range((len(Y)/1024)+1):
            xi,xv,y = self.get_batch(Xi,Xv,Y,1024,i)
            feed_dict = {self.feat_index: xi,
                        self.feat_value: xv,
                        self.label: y[0],
                        self.dropout_keep_fm: self.dropout_fm,
                        self.dropout_keep_deep: self.dropout_deep,
                        self.train_phase: True}
            update, grads = self.sess.run([self.update_all_op, self.grad_op], feed_dict=feed_dict)
            '''if i == len(Y)/1024:
                update, grads = self.sess.run([self.update_all_op, self.grad_op], feed_dict=feed_dict)
                return grads
            else:
                loss, update = self.sess.run([self.loss, self.update_all_op], feed_dict=feed_dict)'''
        feed_dict = {self.feat_index: Xi,
                        self.feat_value: Xv,
                        self.label: Y,
                        self.dropout_keep_fm: self.dropout_fm,
                        self.dropout_keep_deep: self.dropout_deep,
                        self.train_phase: True}
        loss, grads,prediction = self.sess.run([self.loss, self.grad_op,self.out], feed_dict=feed_dict)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(Y, prediction)
        return grads,auc(false_positive_rate, true_positive_rate)
    
    def gradient_transform(self, grads, emb_id_mapping):
        def sum_by_group(values, groups):
            order = np.argsort(groups)
            groups = groups[order]
            values = values[order]
            values.cumsum(axis=0,out=values)
            index = np.ones(len(groups), 'bool')
            index[:-1] = groups[1:] != groups[:-1]
            values = values[index]
            groups = groups[index]
            values[1:] = values[1:] - values[:-1]
            return values, groups

        inv_map = {v: k for k, v in emb_id_mapping.iteritems()}
        for i in  range(len(grads)):
            grad = grads[i]
            if isinstance(grad, IndexedSlicesValue):
                indices = np.vectorize(inv_map.get)(grad.indices)
                values, indices = sum_by_group(grad.values, indices)
                grad = IndexedSlicesValue(values=values, indices=indices, dense_shape=grad.dense_shape)
                grads[i] = grad

        return  grads
