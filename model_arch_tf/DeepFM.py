"""
Tensorflow implementation of DeepFM [1]

Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score,roc_auc_score
from collections import OrderedDict, defaultdict
import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import psycopg2 as p2
from tf_utils import tf_serialize_gradient,tf_deserialize_gradient,tf_deserialize_embedding,tf_serialize_embedding,tf_deserialize_as_nd_weights,tf_serialize_nd_weights,tf_serialize_1d_weights,tf_deserialize_1d_weights
from tensorflow.python.framework.ops import IndexedSlicesValue

class Schema:
    Embed_Model_Table = 'embed_model'
    Dense_Model_Table = 'dense_model'
    Embed_GRADIENT_TABLE = 'embed_gradient_table'
    Dense_GRADIENT_TABLE = 'dense_gradient_table'
    worker = 'model_worker'
    master = 'model_master'

def record(content):
    f = open("/data2/ruike/pg/madlib_model.sql", 'a')
    f.write(content)
    f.close()

class DeepFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size,
                 embedding_size=32, dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 use_fm=True, use_deep=True,
                 loss_type="logloss", eval_metric=accuracy_score,
                 l2_reg=0.0, greater_is_better=True):
        assert (use_fm or use_deep)
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size        # denote as M, size of the feature dictionary
        self.field_size = field_size            # denote as F, size of the feature fields
        self.embedding_size = embedding_size    # denote as K, size of the feature embedding

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                             name="feat_index")  # None * F
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                             name="feat_value")  # None * F
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()

            # model
            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"],
                                                     self.feat_index)  # None * F * K
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)

            # ---------- first order term ----------
            self.y_first_order = tf.nn.embedding_lookup(self.weights["feature_bias"], self.feat_index) # None * F * 1
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)  # None * F
            self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0]) # None * F

            # ---------- second order term ---------------
            # sum_square part
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * K
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # square_sum part
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])  # None * K

            # ---------- Deep component ----------
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size]) # None * (F*K)
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" %i]), self.weights["bias_%d"%i]) # None * layer[i] * 1
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i) # None * layer[i] * 1
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1+i]) # dropout at each Deep layer

            # ---------- DeepFM ----------
            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            elif self.use_deep:
                concat_input = self.y_deep
            self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights["layer_%d"%i])

            # optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95)

            self.train_op = self.optimizer.minimize(self.loss)
            # init
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

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
            dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights


    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z


    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]


    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)


    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.train_op), feed_dict=feed_dict)
        return loss


    def fit(self, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        """
        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)

            # evaluate training and validation datasets
            train_result = self.evaluate(Xi_train, Xv_train, y_train)
            self.train_result.append(train_result)
            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                          % (epoch + 1, train_result, valid_result, time() - t1))
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                          % (epoch + 1, train_result, time() - t1))
            if has_valid and early_stopping and self.training_termination(self.valid_result):
                break

        # fit a few more epoch on train+valid until result reaches the best_train_score
        if has_valid and refit:
            if self.greater_is_better:
                best_valid_score = max(self.valid_result)
            else:
                best_valid_score = min(self.valid_result)
            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]
            Xi_train = Xi_train + Xi_valid
            Xv_train = Xv_train + Xv_valid
            y_train = y_train + y_valid
            for epoch in range(100):
                self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
                total_batch = int(len(y_train) / self.batch_size)
                for i in range(total_batch):
                    Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train,
                                                                 self.batch_size, i)
                    self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                # check
                train_result = self.evaluate(Xi_train, Xv_train, y_train)
                if abs(train_result - best_train_score) < 0.001 or \
                        (self.greater_is_better and train_result > best_train_score) or \
                        ((not self.greater_is_better) and train_result < best_train_score):
                    break


    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                        valid_result[-2] < valid_result[-3] and \
                        valid_result[-3] < valid_result[-4] and \
                        valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                        valid_result[-2] > valid_result[-3] and \
                        valid_result[-3] > valid_result[-4] and \
                        valid_result[-4] > valid_result[-5]:
                    return True
        return False


    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred


    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv, y)
        return self.eval_metric(y, y_pred)

class DeepFM_DA(DeepFM):
    def __init__(self, **kwargs):
        #self.user = 'ruike.xy'
        #self.host = '11.164.101.172'
        #self.dbname = 'driving'
        self.user = 'gpadmin'
        self.host = '172.17.31.87'
        self.dbname = 'gpadmin'
        self.sample_tbl = 'driving'
        self.port = 5432
        #self.total_sample = self._fetch_results("select count(*) from {self.sample_tbl}".format(**locals()))[0][0]
        self.Xi_train, self.Xv_train, self.y_train = list(), list(), list()
        #columns = self._fetch_results("select column_name FROM information_schema.columns WHERE table_name ='{self.sample_tbl}'".format(**locals()))
        #self.xi_index = len(columns) / 2
        #index_columns = columns[self.xi_index+2:]
        feat_dim = 2
        #for column in index_columns:
        #    column = column[0]
         #   num = self._fetch_results("select count(distinct {column}) from {self.sample_tbl}".format(**locals()))[0][0]
          #  feat_dim = feat_dim + num

        kwargs["field_size"] = 39
        if not kwargs["feature_size"]:
            kwargs["feature_size"] = 1
        record("Feat_dim:{}\n".format(feat_dim))
        record("field_size:{}\n".format(39))

        DeepFM.__init__(self, **kwargs)
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


class DeepFM_Master(DeepFM_DA):
    def __init__(self, **kwargs):
        kwargs['feature_size'] = 33000000
        DeepFM_DA.__init__(self, **kwargs)
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

        record("Init DeepFM_Master in DB\n")

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

        '''embeddings = self.sess.run(self.weights['feature_embeddings'])
        embeddings_bias = self.sess.run(self.weights["feature_bias"])
        conn = self._connect_db()
        cursor = conn.cursor()
        for i in range(len(embeddings)):
            embed_weight_serialized = tf_serialize_embedding(embeddings[i])
            embed_bias_weight_serialized = tf_serialize_embedding(embeddings_bias[i])
            shapes = [embeddings[i].shape,embeddings_bias[i].shape]
            sql_insert_embed = "INSERT INTO {} VALUES(%s, %s, %s, %s, {})".format(embed_model_table, self.model_id)
            cursor.execute(sql_insert_embed, (i, p2.Binary(embed_weight_serialized), str(shapes), p2.Binary(embed_bias_weight_serialized)))
        conn.commit()'''
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


    '''def save_embedding(self, weights, embed_id):
        weights_serialized = np.float32(weights[0]).tostring()
        weights_serialized = ",".join(weights_serialized)
        weights_serialized_bias = np.float32(weights[1]).tostring()
        weights_serialized_bias = ",".join(weights_serialized_bias)
        #embed_id = ",".join(embed_id)
        conn = self._connect_db()
        cursor = conn.cursor()
        target = self.model_id+1
        embedding_example = self._fetch_results("SELECT embedding_weight FROM {} WHERE id={}".format(Schema.Embed_Model_Table, 32))
        for i, row in enumerate(embedding_example):
            embed_ex = tf_deserialize_embedding(row[0])
        record("Save embed_model {} in DB, example:{}\n".format(target,embed_ex))
        sql = "Select update_embeddings(ARRAY[{0}],ARRAY[{1}],{2},ARRAY[{3}])".format(p2.Binary(weights_serialized),p2.Binary(weights_serialized_bias),target,embed_id)
        record("Save embed_model {} in DB, example:{}\n".format(target,embed_ex))
        cursor.execute(sql)
        conn.commit()
        embedding_example = self._fetch_results("SELECT embedding_weight FROM {} WHERE id={}".format(Schema.Embed_Model_Table, 32))
        for i, row in enumerate(embedding_example):
            embed_ex = tf_deserialize_embedding(row[0])'''

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

    def get_hot_id(self):
        hot_id_sql = "Select * from hot_key"
        res = self._fetch_results(hot_id_sql)
        hot_id = res[0][0]
        self.hot_id = hot_id

    def hot_key_init(self):
        hot_id = self.hot_id
        hot_id = np.array(hot_id,dtype=int)
        embedding_pulling_sql = "SELECT embedding_weight, embedding_bias FROM {} WHERE id in {}".format(Schema.Embed_Model_Table, tuple(hot_id))
        res = self._fetch_results(embedding_pulling_sql)
        self.update_time = time.time()
        embedding = list()
        embedding_bias = list()
        for i, row in enumerate(res):
            embedding.append(tf_deserialize_embedding(row[0]))
            embedding_bias.append(tf_deserialize_embedding(row[1]))
        variables_ = list()
        variables_.append(np.array(embedding))
        variables_.append(np.array(embedding_bias))
        feed_dict = dict()
        for i, placeholder in enumerate(self.update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.update_ops, feed_dict=feed_dict)

    def hot_key_update(self, new_id):
        pulling_id = list(set(new_id) - set(self.hot_id))
        pulling_id = np.array(pulling_id,dtype=int)
        embedding_pulling_sql = "SELECT embedding_weight, embedding_bias, id FROM {} WHERE id in {}".format(Schema.Embed_Model_Table, tuple(pulling_id))
        res = self._fetch_results(embedding_pulling_sql)
        emb_id_mapping = dict()
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[:2]
        weights_updated = [self.sess.run(v) for v in variables]
        embedding = weights_updated[0]
        embedding_bias = weights_updated[1]
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
        for i, placeholder in enumerate(self.update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.update_ops, feed_dict=feed_dict)
        self.dict_mapping = emb_id_mapping
        return emb_id_mapping

    def apply_grads_loop(self):
        t1 = time.time()
        # while True:
        for i in range(4):
            embed_gradient_table, dense_gradient_table = Schema.Embed_GRADIENT_TABLE, Schema.Dense_GRADIENT_TABLE
            embed_query = '''SELECT worker_id, version FROM {dense_gradient_table} WHERE model_id={self.model_id}'''.format(**locals())
            embed_results = self._fetch_results(embed_query)
            if embed_results:
                record("[Master] Wait for gradient [{} s]".format(round(time.time() - t1, 2)))
                for row in embed_results:
                    worker_id, version = row
                    if not worker_id in self.version.keys():
                        self.version[worker_id] = 0
                    if version == self.version[worker_id]:
                        self.apply_grads_per_worker(worker_id)
                t1 = time.time()

    def apply_embedding_grads_udaf(self):
        t1 = time.time()
        embed_gradient_table = Schema.Embed_GRADIENT_TABLE
        embed_query = '''SELECT worker_id, model_id FROM {embed_gradient_table} WHERE model_id = {self.model_id}'''.format(**locals())
        embed_results = self._fetch_results(embed_query)
        if embed_results:
            record("[Master] Wait for gradient [{} s]\n".format(round(time.time() - t1, 2)))
            for row in embed_results:
                worker_id, model_version = row
                self.apply_embed_grads_per_worker(worker_id, model_version)
        self.model_id = self.model_id  + 1

    def apply_dense_weights(self):
        dense_model_table = Schema.Dense_Model_Table
        dense_fetch = '''SELECT weight,shape FROM dense_model WHERE worker_id = 6'''.format(**locals())
        dense_res = self._fetch_results(dense_fetch)
        shapes = eval(dense_res[0][1])
        record("dense_shapes:{}\n".format(shapes))
        weights = tf_deserialize_as_nd_weights(dense_res[0][0],shapes)
        variables_ = list()
        for v in weights:
            variables_.append(v)
        feed_dict = dict()
        for i, placeholder in enumerate(self.dense_update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.dense_update_ops, feed_dict=feed_dict)
        record("[Master] update Dense at [{}]\n".format(time.time()))
        return

    def dense_average_and_save(self, model_version):
        dense_model_table = Schema.Dense_Model_Table
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        weights_updated = [self.sess.run(v) for v in variables]
        shapes = [v.shape for v in weights_updated[2:]]
        worked_id_list = [0,1,2,3]
        weight_transfer = list()
        for i in worked_id_list:
            dense_fetch = '''SELECT weight FROM {dense_model_table} WHERE model_id={self.model_id} and worker_id = {i}'''.format(**locals())
            weight_transfer.append(tf_deserialize_1d_weights(dense_fetch[0]))
        weights_avg = np.array(weight_transfer).mean(axis = 0)
        serialized_weights = tf_serialize_1d_weights(weights_avg)
        update_sql = '''UPDATE {dense_model_table} SET (model_id, weight) = (self.model_id, %s) where worked_id = 4'''.format(**locals())
        conn = self._connect_db()
        cursor = conn.cursor()
        cursor.execute(update_sql, (p2.Binary(serialized_weights)))
        record("[Master] Save Dense at [{}]\n".format(time.time()))
        conn.commit()
        dense_weights = tf_deserialize_as_nd_weights(serialized_weights,shapes)
        variables_ = list()
        for v in dense_weights:
            variables_.append(v)
        feed_dict = dict()
        for i, placeholder in enumerate(self.dense_update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.dense_update_ops, feed_dict=feed_dict)
        record("[Master] update Dense at [{}]\n".format(time.time()))
        return

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

    def apply_embed_grads_per_worker(self, worker_id, model_version):
        t1 = time.time()
        grads, embed_id_unique = self.pull_embedding_grads(worker_id, model_version)
        t2 = time.time()
        record("[Mater] Pull gradient  [{} s]\n".format(round(t2 - t1,2)))
        self.update_embedding_and_save(worker_id, grads, embed_id_unique)
        t3 = time.time()
        record("[Mater] Update weight [{} s]\n".format(round(t3 - t2,2)))
        record("[Master] [Worker{}] Save embed model with version {}\n".format(worker_id, model_version))

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

    def update_embedding_and_save(self, worker_id, grads, embed_id):
        feed_dict = dict()
        placeholder_count = 0
        for i, grad_ in enumerate(grads):
            if isinstance(grad_, IndexedSlicesValue):
                feed_dict[self.gradient_placehoders[placeholder_count]] = grads[i].values
                placeholder_count = placeholder_count + 1
                feed_dict[self.gradient_placehoders[placeholder_count]] = grads[i].indices
                placeholder_count = placeholder_count + 1
        self.sess.run(self.embed_apply_grad_op, feed_dict=feed_dict)
        self.embedding_dict = self.embedding_dict | set(embed_id)
        #t2 = time.time()
        #embedding_result = self._fetch_results("SELECT embedding_weight, embedding_bias, id FROM {} WHERE id in {}".format(Schema.Embed_Model_Table, tuple(embed_id)))
        #emb_id_mapping = dict()
        #embedding, embedding_bias = list(), list()
        #embed_id_ = list()
        #for i, row in enumerate(embedding_result):
        #    embedding.append(tf_deserialize_embedding(row[0]))
        #    embedding_bias.append(tf_deserialize_embedding(row[1]))
        #    emb_id_mapping[row[2]] = i
        #    embed_id_.append(row[2])
        #t3 = time.time()
        #record("[Master] Pull embedding [{} s]\n".format(round(t3-t2, 2)))
        #variables_ = list()
        #variables_.append(np.array(embedding))
        #variables_.append(np.array(embedding_bias))
#
        #feed_dict = dict()
        #for i, placeholder in enumerate(self.update_placehoders):
        #    feed_dict[placeholder] = variables_[i]
#
        #self.sess.run(self.update_ops, feed_dict=feed_dict)
        #t3 = time.time()
        #feed_dict = dict()
        #placeholder_count = 0
        #for i, grad_ in enumerate(grads):
        #    if isinstance(grad_, IndexedSlicesValue):
        #        indices = np.vectorize(emb_id_mapping.get)(grads[i].indices.astype('int64'))
        #        feed_dict[self.gradient_placehoders[placeholder_count]] = grads[i].values
        #        placeholder_count = placeholder_count + 1
        #        feed_dict[self.gradient_placehoders[placeholder_count]] = indices
        #        placeholder_count = placeholder_count + 1
        #self.sess.run(self.embed_apply_grad_op, feed_dict=feed_dict)
#
        #variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0:2]
        #weights_updated = [self.sess.run(v) for v in variables]
        #for i in range(len(embed_id_)):
        #    self.embedding_cache_dict[embed_id_[i]] = weights_updated[0][i]
        #    self.embedding_bias_cache_dict[embed_id_[i]] = weights_updated[1][i]
        #t4 = time.time()
        #record("[Master] Update weight in master [{} s]\n".format(round(t4-t3, 2)))
        #self.save_embedding(weights_updated, embed_id_)
        #t5 = time.time()
        #record("[Master] [Worker{}] Save embed_weight in master [{} s]\n".format(worker_id,round(t5-t4, 2)))

    def pull_grads(self, worker_id):
        embed_gradient_table, dense_gradient_table = Schema.Embed_GRADIENT_TABLE, Schema.Dense_GRADIENT_TABLE
        version = self.version[worker_id]
        record("[Master] [Worker{}] Start get gradients Version {}".format(worker_id, version))
        query = '''SELECT gradient, shape, auxiliaries FROM {dense_gradient_table} WHERE model_id={self.model_id} AND version={version} AND worker_id={worker_id}'''.format(**locals())
        results = self._fetch_results(query)
        dense_gradient_serialized, shape, auxiliaries = results[0]
        dense_gradients = tf_deserialize_gradient(dense_gradient_serialized, eval(shape), eval(auxiliaries))
        record("[Master] [Worker{}] Receive dense gradients with version {}".format(worker_id, version))

        query = '''SELECT gradient, shape, auxiliaries FROM {embed_gradient_table} WHERE model_id={self.model_id} AND version={version} AND worker_id={worker_id}'''.format(
            **locals())
        results = self._fetch_results(query)
        embed_gradient_serialized, shape, auxiliaries = results[0]
        embed_gradients = tf_deserialize_gradient(embed_gradient_serialized, eval(shape), eval(auxiliaries))
        record("[Master] [Worker{}] Receive embedding gradients with version {}".format(worker_id, version))
        grads = list()
        embed_id = list()
        for e in embed_gradients:
            grads.append(e)
            embed_id.append(e.indices.astype('int64'))
        for d in dense_gradients:
            grads.append(d)

        embed_id_unique = np.unique(np.array(embed_id)).tolist()

        return  grads, embed_id_unique

    def apply_grads_per_worker(self, worker_id):
        t1 = time.time()
        grads, embed_id_unique = self.pull_grads(worker_id)
        t2 = time.time()
        record("[Mater] Pull gradient  [{} s]".format(round(t2 - t1,2)))
        self.update(grads, embed_id_unique)
        t3 = time.time()
        record("[Mater] Update weight [{} s]".format(round(t3 - t2,2)))
        self.version[worker_id] = self.version[worker_id] + 1
        query = "UPDATE {} SET model_version={} WHERE model_id={} AND worker_id={}".format(Schema.Dense_GRADIENT_TABLE, self.version[worker_id], self.model_id,worker_id)
        self._execute(query)
        t4 = time.time()
        record("[Master] [Worker{}] Save model with version {}".format(worker_id, self.version[worker_id]))
        record("[Mater] Deal with worker {} takes {} sec ".format(worker_id, round(t4 - t1,2)))

class DeepFM_Worker(DeepFM_DA):
    def __init__(self, worker_id, **kwargs):
        kwargs['feature_size'] = 300000
        DeepFM_DA.__init__(self, **kwargs)
        self.update_time = None
        self.prune_id = None
        self.dict_mapping = None
        self.hot_id = embedding_cache(cache_capcity=100000,update_staleness=10)
        self.worker_id = worker_id
        self.model_id = self._fetch_results("SELECT max(model_id) from dense_model where worker_id = 6".format(**locals()))[0][0]
        self.version = self.model_id
        #self.sample_tbl = 'criteo_tensor_data'
        #self.total_sample_worker = self._fetch_results_onseg("SELECT count(*) FROM {self.sample_tbl}".format(**locals()))[0][0]
        #self.get_block_info()
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

    def log_variables(self, i):
        variable = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[i]
        variables_value = self.sess.run(variable)
        record(variables_value)

    '''def get_updates(self, serialized_weights):
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[2:]
        variables_value = [self.sess.run(v) for v in variables]
        model_shape = [v.shape.as_list() for v in variables]
        dense_weights = tf_deserialize_as_nd_weights(serialized_weights, model_shape)
        embeddings = self.sess.run(self.weights['feature_embeddings'])
        embeddings_bias = self.sess.run(self.weights["feature_bias"])
        variables_ = list()
        variables_.append(np.array(embeddings))
        variables_.append(np.array(embeddings_bias))
        for v in dense_weights:
            variables_.append(v)
        feed_dict = dict()
        for i, placeholder in enumerate(self.update_placehoders):
            feed_dict[placeholder] = variables_[i]
        self.sess.run(self.update_ops, feed_dict=feed_dict)'''

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
        hot_id = self.hot_id.hot_list
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

    def hot_key_update(self):
        embedding = list()
        embedding_bias = list()
        hot_list = self.hot_id.hot_list
        embed_result = self._fetch_results( "SELECT embedding_weight, embedding_bias, id FROM embed_model WHERE id in {}".format(tuple(hot_list)))
        emb_id_mapping = dict()
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
    
    def pull_dense_grads(self, current_seg_id, target_id):
        dense_gradient_table = Schema.Dense_GRADIENT_TABLE
        check_query_res = False
        while not check_query_res:
            check_query = '''SELECT model_id FROM {dense_gradient_table} WHERE model_id={self.model_id} AND worker_id={target_id}'''.format(**locals())
            check_query_res = self._fetch_results(check_query)
            if check_query_res:
                check_query_res = True
        query = '''SELECT gradient, shape, auxiliaries FROM {dense_gradient_table} WHERE model_id={self.model_id} AND worker_id={target_id}'''.format(**locals())
        results = self._fetch_results(query)
        dense_gradient_serialized, shape, auxiliaries = results[0]
        dense_gradients = tf_deserialize_gradient(dense_gradient_serialized, eval(shape), eval(auxiliaries))
        record("[Worker{}] Receive dense gradients from [Worker{}] with version {}".format(current_seg_id, target_id, self.version))
        return dense_gradients

    def update_dense(self, dense_grads):
        feed_dict = dict()
        placeholder_count = 0
        for i, grad_ in enumerate(dense_grads):
            feed_dict[self.gradient_placehoders[placeholder_count]] = dense_grads[i]
            placeholder_count = placeholder_count + 1
        self.sess.run(self.dense_apply_grad_op, feed_dict=feed_dict)

    def push_embedding_grads_dps(self, batch_id, embedding_grads):
        t1 = time.time()
        embed_grads = embedding_grads
        embed_gradient_table = Schema.Embed_GRADIENT_TABLE
        grads_serialized, shapes, auxiliaries = tf_serialize_gradient(embed_grads)
        sql = "SELECT model_id FROM {embed_gradient_table} WHERE worker_id = {self.worker_id} order by model_id".format(
            **locals())
        result = self._fetch_results(sql)
        target_id = self.model_id + batch_id
        if result == [] or len(result) < 100:
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

    def push_embedding_grads(self, embedding_grads, embed_id_unique):
        t1 = time.time()
        embed_grads = embedding_grads
        embed_gradient_table = Schema.Embed_GRADIENT_TABLE
        grads_serialized, shapes, auxiliaries = tf_serialize_gradient(embed_grads,embed_id_unique)
        sql = "SELECT model_id FROM {embed_gradient_table} WHERE worker_id = {self.worker_id} order by model_id".format(
            **locals())
        result = self._fetch_results(sql)
        if result == [] or len(result) < 10:
            conn = self._connect_db()
            cursor = conn.cursor()
            sql_insert = '''INSERT INTO {embed_gradient_table} (model_id, worker_id, gradient, shape, version, auxiliaries) VALUES ({self.model_id}, {self.worker_id}, %s , %s, {self.version}, %s)'''.format(**locals())
            gs = p2.Binary(grads_serialized)
            cursor.execute(sql_insert, (gs, str(shapes), str(auxiliaries)))
            conn.commit()

        else:
            update_id = result[0][0]
            sql_update = '''UPDATE {embed_gradient_table} SET (model_id, gradient, shape, version, auxiliaries) = ({self.model_id}, %s, %s, {self.version} ,%s) WHERE worker_id={self.worker_id} and model_id = {update_id} '''.format(
                **locals())
            conn = self._connect_db()
            cursor = conn.cursor()
            cursor.execute(sql_update, (p2.Binary(grads_serialized), str(shapes), str(auxiliaries)))
            conn.commit()
        record("[Worker{}] Push embed_gradients with version {} last {}\n".format(self.worker_id, self.version, time.time()-t1))

    def push_dense_grads(self, dense_grads):
        dense_gradient_table = Schema.Dense_GRADIENT_TABLE
        grads_serialized, shapes, auxiliaries = tf_serialize_gradient(dense_grads,None)
        sql = "SELECT version FROM {dense_gradient_table} WHERE worker_id = {self.worker_id}".format(**locals())
        result = self._fetch_results(sql)
        if result == []:
            sql_insert = '''INSERT INTO {dense_gradient_table} (model_id, worker_id, gradient, shape, version, auxiliaries) VALUES ({self.model_id}, {self.worker_id}, %s , %s, {self.version}, %s)'''.format(**locals())
            conn = self._connect_db()
            cursor = conn.cursor()
            cursor.execute(sql_insert, (p2.Binary(grads_serialized), str(shapes), str(auxiliaries)))
            conn.commit()

        else:
            sql_update = '''UPDATE {dense_gradient_table} SET (model_id, gradient, shape, version, auxiliaries) = ({self.model_id}, %s, %s, {self.version} ,%s) WHERE worker_id={self.worker_id}'''.format(**locals())
            conn = self._connect_db()
            cursor = conn.cursor()
            cursor.execute(sql_update, (p2.Binary(grads_serialized), str(shapes), str(auxiliaries)))
            conn.commit()

        record("[Worker{}] Push dense_gradients with version {}\n".format(self.worker_id, self.version))

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

    def pull_dense_weights_udaf(self):
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

    def use_serialize_dense_weights(self,serialized_weights):
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[2:]
        variables_value = [self.sess.run(v) for v in variables]
        model_shape = [v.shape.as_list() for v in variables]
        dense_weights = tf_deserialize_as_nd_weights(serialized_weights,model_shape)
        variables_ = list()
        for v in dense_weights:
            variables_.append(v)
        feed_dict = dict()
        for i, placeholder in enumerate(self.dense_update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.dense_update_ops, feed_dict=feed_dict)

    '''def pull_weights(self, embed_id_unique):
        
        sql_check_version = "SELECT model_version FROM {} WHERE worker_id={} AND model_id={} ".format(Schema.Dense_GRADIENT_TABLE, self.worker_id,self.model_id)
        dense_result = self._fetch_results(sql_check_version)
        t1 = time.time()
        while not (self.init_weight or (dense_result[0][0]==self.version)):
             dense_result = self._fetch_results(sql_check_version)
        record("[Worker{}] Wait for master [{} s]\n".format(self.worker_id, round(time.time() - t1, 2)))
        if self.init_weight:
            self.init_weight = False
        variables_ = list()
        dense_result = self._fetch_results("SELECT weight, shape FROM dense_model WHERE model_id = {}".format(self.model_id))
        shapes_fetch = eval(dense_result[0][1])
        dense_weights = tf_deserialize_as_nd_weights(dense_result[0][0], shapes_fetch)
        embedding = list()
        embedding_bias = list()
        embed_result = self._fetch_results( "SELECT embedding_weight, shape, embedding_bias, id FROM embed_model WHERE id in {} and model_id={}".format(tuple(embed_id_unique), self.model_id))
        emb_id_mapping = dict()
        for i, row in enumerate(embed_result):
            embedding.append(tf_deserialize_embedding(row[0]))
            embedding_bias.append(tf_deserialize_embedding(row[2]))
            emb_id_mapping[row[3]] = i

        variables_.append(np.array(embedding))
        variables_.append(np.array(embedding_bias))
        for v in dense_weights:
            variables_.append(v)
        feed_dict = dict()
        for i, placeholder in enumerate(self.update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.update_ops, feed_dict=feed_dict)

        return emb_id_mapping'''


    def get_block_info(self):
        query = "SELECT (ctid::text::point)[0]::bigint AS block_number, count(*) FROM {self.sample_tbl} where gp_segment_id={self.worker_id} group by block_number;".format(**locals())
        results = self._fetch_results_onseg(query)
        block_info = dict()
        for row in results:
            block_id, count = row
            block_info[int(block_id)] = int(count)

        self.block_info = OrderedDict(sorted(block_info.items()))

    def prune_with_embedding(self, params, n_iter):
        self.prune_id = list()
        target_sparse = params
        adaptive_sparse = target_sparse * (1 - 0.99 ** (n_iter / 100.))
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[:2]
        variables_value = [self.sess.run(v) for v in variables]
        embedding = variables_value[0]
        abs_emb = list()
        for e in embedding:
            abs_emb.append(abs(e).sum())
        embedding_bias = variables_value[1]
        emb_threshold = self.binary_search_threshold(np.array(abs_emb), adaptive_sparse, len(abs_emb))
        for i in range(len(embedding)):
            if abs_emb[i] < emb_threshold:
                self.prune_id.append(i)
        dict_key = list(self.dict_mapping.keys())
        for index in sorted(self.prune_id, reverse=True):
            del dict_key[index]
            del embedding[index]
            del embedding_bias[index]
        temp_dict = dict()
        for i in range(len(dict_key)):
            temp_dict[dict_key[i]] = i
        self.dict_mapping = temp_dict
        variables_ = list()
        variables_.append(np.array(embedding))
        variables_.append(np.array(embedding_bias))
        feed_dict = dict()
        for i, placeholder in enumerate(self.embed_update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.embed_update_ops, feed_dict=feed_dict)

    def binary_search_threshold(self, param, target_percent, total_no):
        l, r = 0., 1e2
        cnt = 0
        mid = 0
        while l < r:
            cnt += 1
            mid = (l + r) / 2
            sparse_items = (abs(param) < mid).sum().item() * 1.0
            sparse_rate = sparse_items / total_no
            if abs(sparse_rate - target_percent) < 0.0001:
                return mid
            elif sparse_rate > target_percent:
                r = mid
            else:
                l = mid
            if cnt > 100:
                break
        return mid

    def check_ctid(self, ctid):
        fid, sid = ctid
        count = 0
        for block_id in self.block_info.keys():
            if block_id < fid:
                count = count + self.block_info[block_id]
            else:
                count = count + sid - 1
                return count

    def get_ctid(self, record_id):
        record_cum = 0
        for block_id in list(self.block_info.keys()):
            if record_cum + self.block_info[block_id] >= record_id + 1:
                res = (block_id, record_id - record_cum + 1)
                assert  self.check_ctid(res) == record_id, "Wrong ctid generated"
                return res
            record_cum = record_cum + self.block_info[block_id]
    
    def prefetch_feature_id(self, batch_size, index):
        time_begin = time.time()
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < self.total_sample_worker else self.total_sample_worker
        start_block = self.get_ctid(start)
        end_block = self.get_ctid(end)
        select_query = "select xi from {self.sample_tbl} where ctid >= '{start_block}' and ctid< '{end_block}'".format(**locals())
        conn = self._connect_seg_db(6000,'127.0.0.1')
        xi = pd.read_sql(select_query,conn)
        assert len(xi) == end - start, "Number of data selected ({}) doesn't match requirement ({})).".format(len(y), end-start)
        record("[Worker{}] prefetch feature id batch {} takes {} sec".format(self.worker_id, index, time.time() - time_begin))
        return xi

    def get_batch_data_block(self, batch_size, index):
        time_begin = time.time()
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < self.total_sample_worker else self.total_sample_worker
        start_block = self.get_ctid(start)
        end_block = self.get_ctid(end)
        select_query = "select xi,xv,y from {self.sample_tbl} where ctid >= '{start_block}' and ctid< '{end_block}'".format(**locals())
        conn = self._connect_seg_db(6000,'127.0.0.1')
        data = pd.read_sql(select_query,conn)
        xi = data['xi'].apply(np.array).values
        xv = data['xv'].apply(np.array).values
        y = data['y'].values
        assert len(y) == end - start, "Number of data selected ({}) doesn't match requirement ({})).".format(len(y), end-start)
        record("[Worker{}] Get batch {} takes {} sec".format(self.worker_id, index, time.time() - time_begin))
        return xi, xv, y

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
        loss, grads = self.sess.run([self.loss, self.grad_op], feed_dict=feed_dict)
        return grads


    def push_dense_weights(self):
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[2:]
        variables_value = [self.sess.run(v) for v in variables]
        flattened_weights = [np.float32(w).tostring() for w in variables_value]
        flattened_weights = "".join(flattened_weights)
        dense_model_table = Schema.Dense_Model_Table
        sql = "SELECT model_id FROM {dense_model_table} WHERE worker_id = {self.worker_id}".format(
            **locals())
        result = self._fetch_results(sql)
        if result == []:
            conn = self._connect_db()
            cursor = conn.cursor()
            sql_insert = '''INSERT INTO {dense_model_table} (model_id, worker_id, weight) VALUES ({self.model_id}, {self.worker_id}, %s)'''.format(**locals())
            cursor.execute(sql_insert, (p2.Binary(flattened_weights),))
            conn.commit()
        else:
            conn = self._connect_db()
            cursor = conn.cursor()
            sql_insert = '''UPDATE {dense_model_table} SET (model_id, weight) = ({self.model_id}, %s) WHERE worker_id = {self.worker_id}'''.format(**locals())
            cursor.execute(sql_insert, (p2.Binary(flattened_weights),))
            conn.commit()


    def push_graident(self, grads):
        embed_grads = grads[0:2]
        dense_grads = grads[2:]
        embed_gradient_table, dense_gradient_table = Schema.Embed_GRADIENT_TABLE, Schema.Dense_GRADIENT_TABLE
        grads_serialized, shapes, auxiliaries = tf_serialize_gradient(embed_grads)
        sql = "SELECT version FROM {embed_gradient_table} WHERE model_id = {self.model_id} AND worker_id = {self.worker_id}".format(
            **locals())
        result = self._fetch_results(sql)
        if result == []:
            conn = self._connect_db()
            cursor = conn.cursor()
            sql_insert = '''INSERT INTO {embed_gradient_table} (model_id, worker_id, gradient, shape, version, auxiliaries) VALUES ({self.model_id}, {self.worker_id}, %s , %s, {self.version}, %s)'''.format(**locals())
            cursor.execute(sql_insert, (p2.Binary(grads_serialized), str(shapes), str(auxiliaries)))
            conn.commit()

        else:
            sql_update = '''UPDATE {embed_gradient_table} SET (gradient, shape, version, auxiliaries) = (%s, %s, {self.version} ,%s) WHERE model_id = {self.model_id} AND worker_id={self.worker_id}'''.format(
                **locals())
            conn = self._connect_db()
            cursor = conn.cursor()
            cursor.execute(sql_update, (p2.Binary(grads_serialized), str(shapes), str(auxiliaries)))
            conn.commit()

        grads_serialized, shapes, auxiliaries = tf_serialize_gradient(dense_grads)
        sql = "SELECT version FROM {dense_gradient_table} WHERE model_id = {self.model_id} AND worker_id = {self.worker_id}".format(**locals())
        result = self._fetch_results(sql)
        if result == []:
            sql_insert = '''INSERT INTO {dense_gradient_table} (model_id, worker_id, gradient, shape, version, auxiliaries) VALUES ({self.model_id}, {self.worker_id}, %s , %s, {self.version}, %s)'''.format(**locals())
            conn = self._connect_db()
            cursor = conn.cursor()
            cursor.execute(sql_insert, (p2.Binary(grads_serialized), str(shapes), str(auxiliaries)))
            conn.commit()

        else:
            sql_update = '''UPDATE {dense_gradient_table} SET (gradient, shape, version, auxiliaries) = (%s, %s, {self.version} ,%s) WHERE model_id = {self.model_id} AND worker_id={self.worker_id}'''.format(**locals())
            conn = self._connect_db()
            cursor = conn.cursor()
            cursor.execute(sql_update, (p2.Binary(grads_serialized), str(shapes), str(auxiliaries)))
            conn.commit()

        record("[Worker{}] Push gradients with version {}\n".format(self.worker_id, self.version))
        self.version = self.version + 1

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

    def run_one_batch(self, batch_id, epoch):
        t1 = time.time()
        Xi_batch, Xv_batch, y_batch = self.get_batch_data_block( self.batch_size, batch_id)
        t2 = time.time()
        record("[Worker{}] Get batch data  [{} s]".format(self.worker_id, round(t2-t1, 2)))
        emd_id_unique = np.unique(np.array(Xi_batch))
        emb_id_mapping = self.pull_weights(emd_id_unique)
        t3 = time.time()
        record("[Worker{}] Pull weight with version {} [{} s]".format(self.worker_id, self.version, round(t3-t2, 2)))
        Xi_batch_local = np.vectorize(emb_id_mapping.get)(Xi_batch)
        grads = self.gradients_compute( Xi_batch_local, Xv_batch, y_batch)
        grads = self.gradient_transform(grads, emb_id_mapping)
        t4 = time.time()
        record("[Worker{}] Compute gradient [{} s]".format(self.worker_id, round(t4-t3, 2)))
        self.push_graident(grads)
        t5 = time.time()
        record("[Worker{}] Push gradient with version {} [{} s]".format(self.worker_id, self.version-1, round(t5-t4, 2)  ))
        train_results = self.evaluate_per_batch(Xi_batch_local, Xv_batch, y_batch)
        t6 = time.time()
        record("[Worker%d] epoch%d batch%d train_results=%.4f [%.1f s]" % (self.worker_id, epoch,batch_id, float(float(train_results)/float(self.batch_size)), t6-t5))
        record("[Worker{}] Time for one batch [{} s]".format(self.worker_id, round(t6-t1, 2)  ))

    def run(self):
        total_batch = int(self.total_sample_worker / self.batch_size)
        record("Total batch:{}".format(total_batch))
        for epoch in range(self.epoch):
            record("[Worker{self.worker_id}] Enter epoch {epoch}".format(**locals()))
            for i in range(total_batch):
                self.run_one_batch(i, epoch)

    def evaluate_per_batch(self, Xi, Xv, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                     self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                     self.train_phase: False}
        batch_out,loss = self.sess.run([self.out,self.loss], feed_dict=feed_dict)
        y_pred = np.reshape(batch_out,newshape = np.array(y).shape)
        metric = self.eval_metric(y, y_pred.round())
        res = []
        res.append(loss)
        res.append(metric)
        return res


class embedding_cache(object):
    def __init__(self, cache_capcity = 100000, update_staleness = 10):
        self.ip = '172.17.31.87'
        self.port = 5432
        self.user = 'gpadmin'
        self.db = 'gpadmin'
        self.capacity = cache_capcity    
        self.cache = OrderedDict()  
        self.hot_list = list()
    
    def init_hot_cache(self, hot_id_list):
        for i in hot_id_list:
            self.cache[i] = 0
        self.hot_list = list(self.cache)
    
    def put_id(self, pulling_id):
        for i in pulling_id:
            self.cache[i] = 1
        if(len(self.cache) > self.capacity):
            self.cache = OrderedDict(sorted(self.cache.items(), key=lambda t: t[1]))
        while(len(self.cache) > self.capacity):
            self.cache.popitem()
        self.hot_list = list(self.cache)
        
