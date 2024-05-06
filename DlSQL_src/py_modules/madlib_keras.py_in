# coding=utf-8
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import datetime
import os
import plpy
import sys
import time

from madlib_keras_helper import *
from madlib_keras_validator import *
from madlib_keras_wrapper import *
from model_arch_info import *
import tensorflow as tf

from madlib_keras_model_selection import ModelSelectionSchema

from internal.db_utils import quote_literal
from utilities.utilities import _assert
from utilities.utilities import add_postfix
from utilities.utilities import is_platform_pg
from utilities.utilities import get_seg_number
from utilities.utilities import madlib_version
from utilities.utilities import unique_string
from utilities.validate_args import get_expr_type
from utilities.validate_args import quote_ident
from utilities.validate_args import input_tbl_valid
from utilities.control import MinWarning

import tensorflow as tf
import utilities.debug as DEBUG

DEBUG.timings_enabled = False
DEBUG.plpy_info_enabled = False

from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *

class GD_STORE:
    SEGMENT_MODEL = 'segment_model'
    AGG_IMAGE_COUNT = 'agg_image_count'

    @staticmethod
    def init(GD, segment_model):
        GD[GD_STORE.SEGMENT_MODEL] = segment_model

    @staticmethod
    def clear(GD):
        del GD[GD_STORE.SEGMENT_MODEL]
        if GD_STORE.AGG_IMAGE_COUNT in GD:
            del GD[GD_STORE.AGG_IMAGE_COUNT]

def get_init_model_and_sess(GD, device_name, gpu_count, segments_per_host,
                               model_architecture, compile_params, custom_function_map):
    # If a live session is present, re-use it. Otherwise, recreate it.

    if GD_STORE.SESS in GD :
        if GD_STORE.SEGMENT_MODEL not in GD:
            plpy.error("Session and model should exist in GD after the first row"
                       "of the first iteration")
        sess = GD[GD_STORE.SESS]
        segment_model = GD[GD_STORE.SEGMENT_MODEL]
        K.set_session(sess)
        record("restore sess\n")
    else:
        sess = get_keras_session(device_name, gpu_count, segments_per_host)
        K.set_session(sess)
        segment_model = init_model(model_architecture, compile_params, custom_function_map)
        GD_STORE.init(GD, sess, segment_model)
        record("init sess\n")
    return segment_model, sess

@MinWarning("warning")
def fit(schema_madlib, source_table, model, model_arch_table,
        model_id, compile_params, fit_params, num_iterations,
        use_gpus, validation_table=None,
        metrics_compute_frequency=None, warm_start=False, name="",
        description="", object_table=None, **kwargs):

    module_name = 'madlib_keras_fit'
    fit_params = "" if not fit_params else fit_params
    _assert(compile_params, "Compile parameters cannot be empty or NULL.")

    input_tbl_valid(source_table, module_name)
    segments_per_host = get_data_distribution_per_segment(source_table)
    use_gpus = use_gpus if use_gpus else False
    if use_gpus:
        accessible_gpus_for_seg = get_accessible_gpus_for_seg(schema_madlib,
                                                              segments_per_host,
                                                              module_name)
    else:
        accessible_gpus_for_seg = get_seg_number()*[0]

    if object_table is not None:
        object_table = "{0}.{1}".format(schema_madlib, quote_ident(object_table))
    fit_validator = FitInputValidator(
        source_table, validation_table, model, model_arch_table, model_id,
        num_iterations, metrics_compute_frequency, warm_start,
        use_gpus, accessible_gpus_for_seg, object_table)

    multi_dep_count = len(fit_validator.dependent_varname)
    src_summary_dict = fit_validator.src_summary_dict
    class_values_colnames = [add_postfix(i, "_class_values") for i in
                             fit_validator.dependent_varname]

    if metrics_compute_frequency is None:
        metrics_compute_frequency = num_iterations

    warm_start = bool(warm_start)

    # The following two times must be recorded together.
    metrics_elapsed_start_time = time.time()
    start_training_time = datetime.datetime.now()
    #TODO add a unit test for this in a future PR
    # save the original value of the env variable so that we can reset it later.
    original_cuda_env = None
    if CUDA_VISIBLE_DEVICES_KEY in os.environ:
        original_cuda_env = os.environ[CUDA_VISIBLE_DEVICES_KEY]

    # Get the serialized master model
    start_deserialization = time.time()
    model_arch, model_weights = get_model_arch_weights(model_arch_table, model_id)

    # The last n layers are the output layers where n is the number of dep vars
    num_classes = get_num_classes(model_arch, multi_dep_count)

    input_shape = get_input_shape(model_arch)
    #fit_validator.validate_input_shapes(input_shape)

    dist_key_col = '0' if is_platform_pg() else DISTRIBUTION_KEY_COLNAME
    gp_segment_id_col = '0' if is_platform_pg() else GP_SEGMENT_ID_COLNAME

    serialized_weights = get_initial_weights(model, model_arch, model_weights,
                                             warm_start, accessible_gpus_for_seg)
    # Compute total images on each segment
    shape_col = fit_validator.dependent_shape_varname[0]
    dist_key_mapping, images_per_seg_train = \
        get_image_count_per_seg_for_minibatched_data_from_db(source_table,
                                                             shape_col)

    if validation_table:
        shape_col = fit_validator.val_dependent_shape_varname[0]
        dist_key_mapping_val, images_per_seg_val = \
            get_image_count_per_seg_for_minibatched_data_from_db(validation_table,
                                                                 shape_col)

    # Construct validation dataset if provided
    validation_set_provided = bool(validation_table)
    validation_metrics = []; validation_loss = []

    # Prepare the SQL for running distributed training via UDA
    compile_params_to_pass = quote_literal(compile_params)
    fit_params_to_pass = quote_literal(fit_params)
    custom_function_map = None

    # If the object_table exists, we read the list of custom
    # function used in the compile_params and map it to their
    # object definition from the object table
    custom_fn_list = get_custom_functions_list(compile_params)
    if object_table is not None:
        custom_function_map = query_custom_functions_map(object_table, custom_fn_list)
    elif len(custom_fn_list) >= 1:
        # Error out if custom_function is called without specifying the object table
        # with the function definition
        plpy.error("Object table not specified for function {0} in compile_params".format(custom_fn_list))

    # Use the smart interface
    if (len(fit_validator.dependent_varname) <= 5 and
        len(fit_validator.independent_varname) <= 5):

        dep_var_array = 5 * ["NULL"]
        indep_var_array = 5 * ["NULL"]

        for counter, var in enumerate(fit_validator.dependent_varname):
            dep_var_array[counter] = var

        for counter, var in enumerate(fit_validator.independent_varname):
            indep_var_array[counter] = var
        mb_dep_var_cols_sql = ', '.join(dep_var_array)
        mb_indep_var_cols_sql = ', '.join(indep_var_array)
    else:

        mb_dep_var_cols_sql = ', '.join(["dependent_var_{0}".format(i)
                                    for i in fit_validator.dependent_varname])
        mb_dep_var_cols_sql = "ARRAY[{0}]".format(mb_dep_var_cols_sql)

        mb_indep_var_cols_sql = ', '.join(["independent_var_{0}".format(i)
                                    for i in fit_validator.independent_varname])
        mb_indep_var_cols_sql = "ARRAY[{0}]".format(mb_indep_var_cols_sql)

    dep_shape_cols_sql = ', '.join(fit_validator.dependent_shape_varname)
    ind_shape_cols_sql = ', '.join(fit_validator.independent_shape_varname)

    run_training_iteration = plpy.prepare("""
        SELECT {schema_madlib}.fit_step(
            {mb_dep_var_cols_sql},
            {mb_indep_var_cols_sql},
            ARRAY[{dep_shape_cols_sql}],
            ARRAY[{ind_shape_cols_sql}],
            $MAD${model_arch}$MAD$::TEXT,
            {compile_params_to_pass}::TEXT,
            {fit_params_to_pass}::TEXT,
            {dist_key_col},
            ARRAY{dist_key_mapping},
            {gp_segment_id_col},
            ARRAY{segments_per_host},
            ARRAY{images_per_seg_train},
            ARRAY{accessible_gpus_for_seg},
            $1,
            $2
        ) AS iteration_result
        FROM {source_table}
        """.format(**locals()), ["bytea", "bytea"])

    # Define the state for the model and loss/metric storage lists
    training_loss, training_metrics, metrics_elapsed_time = [], [], []
    metrics_iters = []

    # get the size of serialized model weights string in KB
    model_size = sys.getsizeof(serialized_weights)/1024.0
    record("model_size: {} \n".format(model_size ))

    # Run distributed training for specified number of iterations
    for i in range(1, num_iterations+1):
        record("begin {} iteration at {}\n".format(i ,time.time()))
        start_iteration = time.time()
        is_final_iteration = (i == num_iterations)

        try:
            record("mdw send model at {}\n".format(time.time()))
            serialized_weights = plpy.execute(run_training_iteration,
                                            [serialized_weights, custom_function_map]
                                            )[0]['iteration_result']
            record("mdw finish fit_step at {}\n".format(time.time()))
            model_size = sys.getsizeof(serialized_weights)/1024.0
            record("model_size: {} \n".format(model_size ))
        except plpy.SPIError as e:
            msg = e.message
            if 'TransAggDetail' in msg:
                e.message, detail = msg.split('TransAggDetail')
            elif 'MergeAggDetail' in msg:
                e.message, detail = msg.split('MergeAggDetail')
            elif 'FinalAggDetail' in msg:
                e.message, detail = msg.split('FinalAggDetail')
            else:
                raise e
            # Extract Traceback from segment, add to
            #  DETAIL of error message on coordinator
            e.args = (e.message,)
            spidata = list(e.spidata)
            spidata[1] = detail
            e.spidata = tuple(spidata)
            raise e

        end_iteration = time.time()
        info_str = "\tTime for training in iteration {0}: {1} sec".format(i,
            end_iteration - start_iteration)

        if should_compute_metrics_this_iter(i, metrics_compute_frequency,
                                            num_iterations):
            """
            If there is no validation dataset, we should clear the session/gd at
            the last call to train evaluate. Otherwise clear it at the last call
            to validation evaluate
            """
            should_clear_session = False
            if not validation_set_provided:
                should_clear_session = is_final_iteration

            record("mdw begin compute_loss_and_metrics for train set at {}\n".format(time.time()))
            compute_out = compute_loss_and_metrics(schema_madlib, source_table,
                                                   fit_validator.dependent_varname,
                                                   fit_validator.independent_varname,
                                                   compile_params_to_pass,
                                                   fit_params_to_pass,
                                                   model_arch,
                                                   serialized_weights, use_gpus,
                                                   accessible_gpus_for_seg,
                                                   segments_per_host,
                                                   dist_key_mapping,
                                                   images_per_seg_train,
                                                   training_metrics,
                                                   training_loss,
                                                   should_clear_session,
                                                   custom_function_map)
            record("mdw finish compute_loss_and_metrics for train set at {}\n".format(time.time()))
            metrics_iters.append(i)
            compute_time, compute_metrics, compute_loss = compute_out
            info_str = get_evaluate_info_msg(i, info_str, compute_out, True)
            if validation_set_provided:
                # Compute loss/accuracy for validation data.
                record("mdw begin compute_loss_and_metrics for vaild set at {}\n".format(time.time()))
                val_compute_out = compute_loss_and_metrics(schema_madlib,
                                                           validation_table,
                                                           fit_validator.val_dependent_varname,
                                                           fit_validator.val_independent_varname,
                                                           compile_params_to_pass,
                                                           fit_params_to_pass,
                                                           model_arch,
                                                           serialized_weights,
                                                           use_gpus,
                                                           accessible_gpus_for_seg,
                                                           segments_per_host,
                                                           dist_key_mapping_val,
                                                           images_per_seg_val,
                                                           validation_metrics,
                                                           validation_loss,
                                                           is_final_iteration,
                                                           custom_function_map)
                info_str = get_evaluate_info_msg(i, info_str, val_compute_out,
                                                 False)
                record("mdw finish compute_loss_and_metrics for vaild set at {}\n".format(time.time()))

            metrics_elapsed_end_time = time.time()
            metrics_elapsed_time.append(
                metrics_elapsed_end_time-metrics_elapsed_start_time)
        plpy.info("\n"+info_str)
        record("finish {} iteration at {}\n".format(i ,time.time()))
    end_training_time = datetime.datetime.now()

    version = madlib_version(schema_madlib)
    norm_const = src_summary_dict['normalizing_const']
    dep_vartype = src_summary_dict['dependent_vartype']
    dependent_varname = src_summary_dict['dependent_varname']
    independent_varname = src_summary_dict['independent_varname']

    dep_name_list = ', '.join([quote_literal(i) for i in dependent_varname])
    ind_name_list = ', '.join([quote_literal(i) for i in independent_varname])

    # Define some constants to be inserted into the summary table.
    model_type = "madlib_keras"
    metrics_list = get_metrics_from_compile_param(compile_params)
    is_metrics_specified = True if metrics_list else False
    metrics_type = 'ARRAY{0}'.format(metrics_list) if is_metrics_specified else 'NULL'
    metrics_iters = metrics_iters if metrics_iters else 'NULL'
    loss_type = get_loss_from_compile_param(compile_params)

    # We always compute the training loss and metrics, at least once.
    training_metrics_final, training_metrics = get_metrics_sql_string(
        training_metrics, is_metrics_specified)
    training_loss_final, training_loss = get_metrics_sql_string(
        training_loss, True)

    # Validation loss and metrics are computed only if validation_table
    # is provided.
    if validation_set_provided:
        validation_metrics_final, validation_metrics = get_metrics_sql_string(
            validation_metrics, is_metrics_specified)
        validation_loss_final, validation_loss = get_metrics_sql_string(validation_loss)
        # Must quote the string before inserting to table. Explicitly
        # quoting it here since this can also take a NULL value, done
        # in the else part.
        validation_table = quote_literal(validation_table)
    else:
        validation_metrics = validation_loss = 'NULL'
        validation_metrics_final = validation_loss_final = 'NULL'
        validation_table = 'NULL'

    object_table = quote_literal(object_table) if object_table is not None else 'NULL'
    class_values_colnames = ' , '.join(class_values_colnames)
    if warm_start:
        plpy.execute("DROP TABLE {0}, {1}".format
                     (model, fit_validator.output_summary_model_table))
    create_output_summary_table = plpy.prepare("""
        CREATE TABLE {output_summary_model_table} AS
        SELECT
            $MAD${source_table}$MAD$::TEXT AS source_table,
            $MAD${model}$MAD$::TEXT AS model,
            ARRAY[{dep_name_list}]::TEXT[] AS dependent_varname,
            ARRAY[{ind_name_list}]::TEXT[] AS independent_varname,
            $MAD${model_arch_table}$MAD$::TEXT AS model_arch_table,
            {model_id}::INTEGER AS {model_id_colname},
            $1 AS compile_params,
            $2 AS fit_params,
            {num_iterations}::INTEGER AS num_iterations,
            {validation_table}::TEXT AS validation_table,
            {object_table}::TEXT AS object_table,
            {metrics_compute_frequency}::INTEGER AS metrics_compute_frequency,
            $3 AS name,
            $4 AS description,
            '{model_type}'::TEXT AS model_type,
            {model_size}::DOUBLE PRECISION AS model_size,
            '{start_training_time}'::TIMESTAMP AS start_training_time,
            '{end_training_time}'::TIMESTAMP AS end_training_time,
            $5 AS metrics_elapsed_time,
            '{version}'::TEXT AS madlib_version,
            ARRAY{num_classes}::INTEGER[] AS num_classes,
            ARRAY{dep_vartype}::TEXT[] AS {dependent_vartype_colname},
            {norm_const}::{FLOAT32_SQL_TYPE} AS {normalizing_const_colname},
            {metrics_type}::TEXT[] AS metrics_type,
            '{loss_type}'::TEXT AS loss_type,
            {training_metrics_final}::DOUBLE PRECISION AS training_metrics_final,
            {training_loss_final}::DOUBLE PRECISION AS training_loss_final,
            {training_metrics}::DOUBLE PRECISION[] AS training_metrics,
            {training_loss}::DOUBLE PRECISION[] AS training_loss,
            {validation_metrics_final}::DOUBLE PRECISION AS validation_metrics_final,
            {validation_loss_final}::DOUBLE PRECISION AS validation_loss_final,
            {validation_metrics}::DOUBLE PRECISION[] AS validation_metrics,
            {validation_loss}::DOUBLE PRECISION[] AS validation_loss,
            ARRAY{metrics_iters}::INTEGER[] AS metrics_iters,
            {class_values_colnames}
        FROM {source_summary_table}
        """.format(output_summary_model_table=fit_validator.output_summary_model_table,
                   dependent_vartype_colname=DEPENDENT_VARTYPE_COLNAME,
                   normalizing_const_colname=NORMALIZING_CONST_COLNAME,
                   FLOAT32_SQL_TYPE = FLOAT32_SQL_TYPE,
                   model_id_colname = ModelArchSchema.MODEL_ID,
                   source_summary_table=fit_validator.source_summary_table,
                   **locals()),
                   ["TEXT", "TEXT", "TEXT", "TEXT", "DOUBLE PRECISION[]"])
    plpy.execute(create_output_summary_table,
                 [compile_params, fit_params, name,
                  description, metrics_elapsed_time])

    plpy.execute("""
        CREATE TABLE {0}
        (model_weights bytea,
        {1} json)""".format(model, ModelArchSchema.MODEL_ARCH))
    insert_output_table = plpy.prepare("""
        INSERT INTO {0} SELECT model_weights, {1}
        FROM (VALUES($1, $2))t(model_weights, {1})
        """.format(model, ModelArchSchema.MODEL_ARCH), ["bytea", "json"])
    plpy.execute(insert_output_table, [serialized_weights, model_arch])

    #TODO add a unit test for this in a future PR
    reset_cuda_env(original_cuda_env)


def get_evaluate_info_msg(i, info_str, compute_out, is_train):
    compute_time, compute_metrics, compute_loss = compute_out
    if is_train:
        label = "Training"
    else:
        label = "Validation"
    info_str += "\n\tTime for evaluating {0} dataset in " \
                "iteration {1}: {2} sec\n".format(label.lower(), i, compute_time)
    info_str += "\t{0} set metric after iteration {1}: {2}\n".format(
        label, i, compute_metrics)
    info_str += "\t{0} set loss after iteration {1}: {2}".format(
        label, i, compute_loss)
    return info_str

from tensorflow.keras import backend  as KK

class MyLayer(Layer):
    def __init__(self, output_dim= 1, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1].value, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x):
        return KK.dot(x, self.kernel)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    def get_config(self):
        config = super(MyLayer,self).get_config()
        config['output_dim'] =  self.output_dim# say self. _localization_net  if you store the argument in __init__
        return config


def get_initial_weights(model_table, model_arch, serialized_weights, warm_start,
                        accessible_gpus_for_seg, mst_filter=''):
    """
        If warm_start is True, return back initial weights from model table.
        If warm_start is False, first try to get the weights from model_arch
        table, if no weights are defined there, randomly initialize it using
        keras.
        We also need to set the cuda environment variable based on the platform.
        1. For postgres, if user specifies use_gpus=False which means they want
        to use CPU, then we have to set CUDA_VISIBLE_DEVICES to -1 to disable gpu.
        Otherwise model.get_weights() will use gpu if available.

        2. For gpdb, we want to disable gpu on gpdb's master node because GPUs
        will only be used for segment nodes.
        @args:
            @param model_table: Output model table passed in to fit.
            @param model_arch: Dict containing model architecture info.
            @param warm_start: Boolean flag indicating warm start or not.
    """
    if is_platform_pg():
        # Use GPU's if they are enabled
        _ = get_device_name_and_set_cuda_env(accessible_gpus_for_seg[0], None)
    else: # gpdb
        # We are on master, so never use GPU's
        _ = get_device_name_and_set_cuda_env(0, None)

    if warm_start:
        serialized_weights = plpy.execute("""
            SELECT model_weights FROM {model_table} {mst_filter} LIMIT 1
        """.format(**locals()))[0]['model_weights']
    else:
        if not serialized_weights:
            model = model_from_json(model_arch, custom_objects={'MyLayer': MyLayer})
            record("start serialized_weights at {}\n".format(time.time()))
            serialized_weights = madlib_keras_serializer.serialize_nd_weights(
                model.get_weights())
            record("finish serialized_weights at {}\n".format(time.time()))
    return serialized_weights

def get_source_summary_table_dict(source_summary_table):
    source_summary = plpy.execute("""
            SELECT *
            FROM {0}
        """.format(source_summary_table))[0]

    return source_summary

def compute_loss_and_metrics(schema_madlib, table, dependent_varname,
                             independent_varname, compile_params, fit_params,
                             serialized_weights, use_gpus,
                             accessible_gpus_for_seg, segments_per_host,
                             dist_key_mapping, images_per_seg_val, metrics_list,
                             loss_list, should_clear_session, custom_fn_map,
                             model_table=None, mst_key=None):
    """
    Compute the loss and metric using a given model (serialized_weights) on the
    given dataset (table.)
    """
    start_val = time.time()
    evaluate_result = get_loss_metric_from_keras_eval_ctq(schema_madlib, table,
                                                      dependent_varname,
                                                      independent_varname,
                                                      compile_params,
                                                      fit_params,
                                                      serialized_weights,
                                                      use_gpus,
                                                      accessible_gpus_for_seg,
                                                      segments_per_host,
                                                      dist_key_mapping,
                                                      images_per_seg_val,
                                                      should_clear_session,
                                                      custom_fn_map, model_table,
                                                      mst_key)
    end_val = time.time()
    loss = evaluate_result[0]
    metric = evaluate_result[1]
    metrics_list.append(metric)
    loss_list.append(loss)
    return end_val - start_val, metric, loss

def should_compute_metrics_this_iter(curr_iter, metrics_compute_frequency,
                                     num_iterations):
    """
    Check if we want to compute loss/accuracy for the current iteration
    :param curr_iter:
    :param metrics_compute_frequency:
    :param num_iterations:
    :return: Returns a boolean
            return TRUE, if it is the last iteration, or if metrics_compute_frequency
            iterations have elapsed since the last time it was computed.
            return FALSE otherwise.
    """
    # Compute loss/accuracy every metrics_compute_frequency'th iteration,
    # and also for the last iteration.
    return 1==0

def init_model(model_architecture, compile_params, custom_function_map):
    """
        Should only be called at the first row of first iteration.
    """
    segment_model = model_from_json(model_architecture, custom_objects={'MyLayer': MyLayer})
    compile_model(segment_model, compile_params, custom_function_map)
    return segment_model

def fit_transition_wide(state, dependent_var1, dependent_var2, dependent_var3,
                   dependent_var4, dependent_var5, independent_var1,
                   independent_var2, independent_var3, independent_var4,
                   independent_var5, dependent_var_shape,
                   independent_var_shape, model_architecture,
                   compile_params, fit_params, dist_key, dist_key_mapping,
                   current_seg_id, segments_per_host, images_per_seg,
                   accessible_gpus_for_seg, prev_serialized_weights,
                   is_multiple_model=False, custom_function_map=None, **kwargs):

    if not independent_var1 or not dependent_var1:
        return state
    dependent_var = [dependent_var1, dependent_var2, dependent_var3,
                        dependent_var4, dependent_var5]
    independent_var = [independent_var1, independent_var2, independent_var3,
                        independent_var4, independent_var5]

    dependent_var = [i for i in dependent_var if i is not None]
    independent_var = [i for i in independent_var if i is not None]

    return fit_transition(state, dependent_var, independent_var, dependent_var_shape,
                   independent_var_shape, model_architecture,
                   compile_params, fit_params, dist_key, dist_key_mapping,
                   current_seg_id, segments_per_host, images_per_seg,
                   accessible_gpus_for_seg, prev_serialized_weights,
                   is_multiple_model, custom_function_map, **kwargs)

def record(content):
    f = open("/data2/ruike/pg/madlib_model.sql", 'a')
    f.write(content)
    f.close()

def fit_transition(state, dependent_var, independent_var, dependent_var_shape,
                   independent_var_shape, model_architecture,
                   compile_params, fit_params, dist_key, dist_key_mapping,
                   current_seg_id, segments_per_host, images_per_seg,
                   accessible_gpus_for_seg, prev_serialized_weights,
                   is_multiple_model=False, custom_function_map=None, **kwargs):
    """
    This transition function is common for madlib_keras_fit() and
    madlib_keras_fit_multiple_model(). The important difference between
    these two calls is the way tensorflow/keras sessions and GD gets used.
    For madlib_keras_fit_multiple_model,
        a. We create a tensorflow session per hop and store it in GD alongwith
        the model and clear both GD and the session at the end of each
        hop.
    For madlib_keras_fit,
        b. We create only one tensorflow session for both fit and eval transition
        functions and store it in GD. This session gets reused by both fit and eval
        and only gets cleared in eval transition at the last row of the last iteration.

    """
    if not dependent_var_shape[0] or not independent_var_shape[0]\
        or dependent_var[0] is None or independent_var[0] is None:
            plpy.error("fit_transition called with no data")

    if not prev_serialized_weights or not model_architecture:
        return state

    current_seg_id = dist_key_mapping.index(dist_key)
    record("seg {} enter fit_trans at {}\n".format(current_seg_id ,time.time()))
    GD = kwargs['GD']

    trans_enter_time = time.time()

    device_name = get_device_name_and_set_cuda_env(accessible_gpus_for_seg[current_seg_id], current_seg_id)

    segment_model, sess = get_init_model_and_sess(GD, device_name,
        accessible_gpus_for_seg[current_seg_id],
        segments_per_host[current_seg_id],
        model_architecture, compile_params,
        custom_function_map)

    if GD_STORE.AGG_IMAGE_COUNT in GD:
        record("seg {} restore model at {}\n".format(current_seg_id, time.time()))
        agg_image_count = GD[GD_STORE.AGG_IMAGE_COUNT]
    else:
        record("seg {} receive model at {}\n".format(current_seg_id, time.time()))
        agg_image_count = 0
        GD[GD_STORE.AGG_IMAGE_COUNT] = agg_image_count
        set_model_weights(segment_model, prev_serialized_weights)
    record("seg {} finish set model at {}\n".format(current_seg_id, time.time()))

    x_train = []
    y_train = []
    # Prepare the data
    for counter, shape in enumerate(independent_var_shape):
        x_train.append(np_array_float32(independent_var[counter], shape))

    for counter, shape in enumerate(dependent_var_shape):
        y_train.append(np_array_int16(dependent_var[counter], shape))

    # Fit segment model on data
    #TODO consider not doing this every time
    fit_params = parse_and_validate_fit_params(fit_params, current_seg_id)
    x_train_tmp = x_train[0].T
    X = [np.array(x_train_tmp[i, :]) for i in range(x_train_tmp.shape[0])]
    f = open("/data2/ruike/tmp.txt", 'a')
    f.write("test !\n")
    f.write(str(len(y_train)))
    f.write(str(y_train[0].shape))
    f.write(str(len(x_train)))
    f.write(str(x_train[0].shape))
    f.write(str(y_train[0][:,0].reshape(-1,1).shape))
    f.write(str(X[0].shape))
    f.write("Train model at {}\n".format(time.time()))
    f.close()
    record("seg {} train model at {}\n".format(current_seg_id, time.time()))
    segment_model.fit(X, y_train[0][:,0].reshape(-1, 1), **fit_params)
    # Aggregating number of images, loss and accuracy
    record("seg {} finish train model at {}\n".format(current_seg_id, time.time()))
    agg_image_count += len(x_train[0])
    GD[GD_STORE.AGG_IMAGE_COUNT] = agg_image_count
    total_images = get_image_count_per_seg_from_array(dist_key_mapping.index(dist_key),
                                                      images_per_seg)
    is_last_row = agg_image_count == total_images
    return_state = get_state_to_return(segment_model, is_last_row, is_multiple_model,
                                       agg_image_count, total_images)

    if is_last_row:
        del GD[GD_STORE.AGG_IMAGE_COUNT]  # Must be reset after each pass through images
        if is_multiple_model:
            GD_STORE.clear(GD)
            clear_keras_session(sess)

    trans_exit_time = time.time()
    DEBUG.plpy.info("|_fit_transition_time_|{}|".format(trans_exit_time - trans_enter_time))
    record("seg {} leave fit_trans at {}\n".format(current_seg_id ,time.time()))
    record("seg {} fit_trans last {}\n".format(current_seg_id ,trans_exit_time - trans_enter_time))
    return return_state

def fit_multiple_transition_caching(dependent_var, independent_var, dependent_var_shape,
                             independent_var_shape, model_architecture,
                             compile_params, fit_params, dist_key, dist_key_mapping,
                             current_seg_id, segments_per_host, images_per_seg,
                             accessible_gpus_for_seg, serialized_weights,
                             is_final_training_call, custom_function_map=None, **kwargs):
    """
    This transition function is called when caching is called for
    madlib_keras_fit_multiple_model().
    The input params: dependent_var, independent_var,
    dependent_var_shape and independent_var_shape are passed
    in as None for all hops except the very first hop
    Some things to note in this function are:
    - weights can be passed in as None for the very first hop
      and the final training call.  (This can only happen if
      num msts < num segs)
    - x_train, y_train and cache_set is cleared from GD for
      is_final_training_call = True
    """
    GD = kwargs['GD']

    trans_enter_time = time.time()

    if GD_STORE.AGG_IMAGE_COUNT in GD:
        agg_image_count = GD[GD_STORE.AGG_IMAGE_COUNT]
    else:
        agg_image_count = 0
        GD[GD_STORE.AGG_IMAGE_COUNT] = agg_image_count

    # Prepare the data
    if not dependent_var_shape[0] or not independent_var_shape[0] \
        or dependent_var[0] is None or independent_var[0] is None:
        if 'x_train' not in GD or 'y_train' not in GD:
            plpy.error("cache not populated properly.")
        is_last_row = True
        total_images = None
    else:
        if 'x_train' not in GD or 'y_train' not in GD:
            GD['x_train'] = list()
            GD['y_train'] = list()

        #TODO: Fix the [0] for multi io
        agg_image_count += independent_var_shape[0][0]

        GD[GD_STORE.AGG_IMAGE_COUNT] = agg_image_count
        total_images = get_image_count_per_seg_from_array(
            dist_key_mapping.index(dist_key), images_per_seg
        )
        is_last_row = agg_image_count == total_images
        x_train_current = np_array_float32(independent_var[0], independent_var_shape[0])
        y_train_current = np_array_int16(dependent_var[0], dependent_var_shape[0])
        GD['x_train'].append(x_train_current)
        GD['y_train'].append(y_train_current)

    # Passed in weights can be None. Irrespective of the weights, we want to populate the cache for the very first hop.
    # But if the weights are None, we do not want to set any model. So early return in that case
    if serialized_weights is None:
        if is_final_training_call:
            del GD[GD_STORE.AGG_IMAGE_COUNT]
            del GD['x_train']
            del GD['y_train']
        return None

    segment_model = None
    sess = None

    if is_last_row:
        device_name = get_device_name_and_set_cuda_env(accessible_gpus_for_seg[current_seg_id], current_seg_id)
        segment_model, sess = get_init_model_and_sess(GD, device_name,
                                                      accessible_gpus_for_seg[current_seg_id],
                                                      segments_per_host[current_seg_id],
                                                      model_architecture, compile_params,
                                                      custom_function_map)

        set_model_weights(segment_model, serialized_weights)
        fit_params = parse_and_validate_fit_params(fit_params, current_seg_id)

        for i in range(len(GD['x_train'])):
            # Fit segment model on data
            segment_model.fit(GD['x_train'][i], GD['y_train'][i], **fit_params)

    return_state = get_state_to_return(segment_model, is_last_row, True,
                                       agg_image_count)

    if is_last_row:
        GD_STORE.clear(GD)
        clear_keras_session(sess)
        if is_final_training_call:
            if GD_STORE.AGG_IMAGE_COUNT in GD:
                del GD[GD_STORE.AGG_IMAGE_COUNT]
            del GD['x_train']
            del GD['y_train']

    trans_exit_time = time.time()
    DEBUG.plpy.info("|_fit_multiple_transition_caching_time_|{}|".format(trans_exit_time - trans_enter_time))

    return return_state

def get_state_to_return(segment_model, is_last_row, is_multiple_model, agg_image_count,
                        total_images=None):
    """
    1. For both model averaging fit_transition and fit multiple transition, the
    state only needs to have the image count except for the last row.
    2. For model averaging fit_transition, the last row state must always contains
    the image count as well as the model weights. This state then gets passed to the
    merge and final functions.
    3. For fit multiple transition, the last row state only needs the model
    weights. This state is the output of the UDA for that hop. We don't need
    the image_count here because unlike model averaging, model hopper does
    not have a merge/final function and there is no need to average the weights
    based on the image count.
    :param segment_model: cached model for that segment
    :param is_last_row: boolean to indicate if last row for that hop
    :param is_multiple_model: boolean
    :param agg_image_count: aggregated image count per hop
    :param total_images: total images per segment (only used for madlib_keras_fit() )
    :return:
    """
    if is_multiple_model:
        if is_last_row:
            updated_model_weights = segment_model.get_weights()
            new_state = madlib_keras_serializer.serialize_nd_weights(updated_model_weights)
        else:
            new_state = None
    elif is_last_row:
        updated_model_weights = segment_model.get_weights()
        updated_model_weights = [total_images * w for w in updated_model_weights]
        new_state = madlib_keras_serializer.serialize_state_with_nd_weights(
            agg_image_count, updated_model_weights)
    else:
        new_state = float(agg_image_count)

    return new_state

def get_grads_to_return(is_last_row, is_multiple_model, agg_image_count,
                        total_images=None):
    """
    1. For both model averaging fit_transition and fit multiple transition, the
    state only needs to have the image count except for the last row.
    2. For model averaging fit_transition, the last row state must always contains
    the image count as well as the model weights. This state then gets passed to the
    merge and final functions.
    3. For fit multiple transition, the last row state only needs the model
    weights. This state is the output of the UDA for that hop. We don't need
    the image_count here because unlike model averaging, model hopper does
    not have a merge/final function and there is no need to average the weights
    based on the image count.
    :param segment_model: cached model for that segment
    :param is_last_row: boolean to indicate if last row for that hop
    :param is_multiple_model: boolean
    :param agg_image_count: aggregated image count per hop
    :param total_images: total images per segment (only used for madlib_keras_fit() )
    :return:
    """
    if is_multiple_model:
        if is_last_row:
            new_state = None
        else:
            new_state = None
    elif is_last_row:
        state = [np.array([agg_image_count])]
        new_state = np.float32(state).tostring()
    else:
        new_state = float(agg_image_count)
    return new_state

def fit_merge(state1, state2, **kwargs):
    record("mdw enter fit_merge at {}\n".format(time.time()))

    # Return if called early
    if not state1 or not state2:
        return state1 or state2

    # Deserialize states
    image_count1, weights1 = madlib_keras_serializer.deserialize_as_image_1d_weights(state1)
    image_count2, weights2 = madlib_keras_serializer.deserialize_as_image_1d_weights(state2)

    # Compute total image counts
    image_count = (image_count1 + image_count2) * 1.0

    # Aggregate the weights
    total_weights = weights1 + weights2

    # Return the merged state
    record("mdw finish fit_merge at {}\n".format(time.time()))
    return madlib_keras_serializer.serialize_state_with_1d_weights(
        image_count, total_weights)

def fit_final(state, **kwargs):
    record("mdw enter fit_final at {}\n".format(time.time()))
    # Return if called early
    if not state:
        return state
    image_count, weights = madlib_keras_serializer.deserialize_as_image_1d_weights(state)
    if image_count == 0:
        plpy.error("fit_final: Total images processed is 0")

    # Averaging the weights
    weights /= image_count
    record("image_count:{}\n".format(image_count))
    record("mdw start serialize_nd_weights at {}\n".format(time.time()))
    res = madlib_keras_serializer.serialize_nd_weights(weights)
    record("mdw finish serialize_nd_weights at {}\n".format(time.time()))

    model_size = sys.getsizeof(weights)/1024.0
    record("model_size: {} \n".format(model_size ))

    record("mdw finish fit_final at {}\n".format(time.time()))
    return res

def fit_grads_merge(state1, state2, **kwargs):
    record("mdw enter fit_merge_grads at {}\n".format(time.time()))

    # Return if called early
    if not state1 or not state2:
        return state1 or state2

    # Deserialize states
    image_count1 = np.fromstring(state1, dtype=np.float32)
    image_count2 = np.fromstring(state2, dtype=np.float32)

    # Compute total image counts
    image_count = image_count1 + image_count2

    # Aggregate the weights
    state = [np.array([image_count])]
    new_state = np.float32(state).tostring()

    # Return the merged state
    record("mdw finish fit_merge_grads at {}\n".format(time.time()))
    return new_state

def fit_grads_final(state, **kwargs):
    record("mdw enter fit_final_grads at {}\n".format(time.time()))
    # Return if called early
    if not state:
        return state
    image_count = np.fromstring(state, dtype=np.float32)
    '''if image_count == 0:
        plpy.error("fit_final: Total images processed is 0")'''
    image_count = image_count / 6
    state = tf_serialize_nd_weights(image_count)
    # Averaging the weights
    record("image_count:{}\n".format(image_count))
    record("mdw finish fit_final_grads at {}\n".format(time.time()))
    return state


def evaluate(schema_madlib, model_table, test_table, output_table,
             use_gpus, mst_key, **kwargs):

    module_name = 'madlib_keras_evaluate'
    is_mult_model = mst_key is not None
    test_summary_table = None
    if test_table:
        test_summary_table = add_postfix(test_table, "_summary")
    model_summary_table = None
    if model_table:
        model_summary_table = add_postfix(model_table, "_summary")

    mult_where_clause = ""
    input_tbl_valid(model_table, module_name)
    if is_mult_model:
        mult_where_clause = "WHERE mst_key = {0}".format(mst_key)
        model_summary_table = create_summary_view(module_name, model_table, mst_key)

    validate_evaluate(module_name, model_table, model_summary_table, test_table, test_summary_table, output_table, is_mult_model)

    segments_per_host = get_data_distribution_per_segment(test_table)
    if use_gpus:
        accessible_gpus_for_seg = get_accessible_gpus_for_seg(schema_madlib,
                                                              segments_per_host,
                                                              module_name)
    else:
        accessible_gpus_for_seg = get_seg_number()*[0]

    model_weights_query = "SELECT model_weights, model_arch FROM {0} {1}".format(
        model_table, mult_where_clause)

    res = plpy.execute(model_weights_query)[0]
    _assert(res, "{0}: The model does not exist.")
    model_weights = res['model_weights']
    model_arch = res['model_arch']

    input_shape = get_input_shape(model_arch)

    model_summary_dict = get_source_summary_table_dict(model_summary_table)
    # independent_varname = model_summary_dict['independent_varname']
    # ind_shape_cols = [add_postfix(i, "_shape") for i in independent_varname]

    dep_varname = model_summary_dict['dependent_varname']
    indep_varname = model_summary_dict['independent_varname']

    InputValidator.validate_input_shape(
        test_table, indep_varname, input_shape, 2, True)

    compile_params_query = "SELECT compile_params, fit_params, metrics_type, object_table FROM {0}".format(model_summary_table)
    res = plpy.execute(compile_params_query)[0]
    metrics_type = res['metrics_type']
    compile_params = quote_literal(res['compile_params'])
    fit_params = quote_literal(res['fit_params'])
    object_table = res['object_table']
    loss_type = get_loss_from_compile_param(res['compile_params'])
    custom_function_map = None
    if object_table is not None:
        custom_fn_list = get_custom_functions_list(res['compile_params'])
        custom_function_map = query_custom_functions_map(object_table, custom_fn_list)

    shape_col = add_postfix(dep_varname[0], "_shape")
    dist_key_mapping, images_per_seg = \
        get_image_count_per_seg_for_minibatched_data_from_db(test_table, shape_col)

    loss_metric = \
        get_loss_metric_from_keras_eval(
            schema_madlib, test_table, dep_varname, indep_varname, compile_params, fit_params, model_arch,
            model_weights, use_gpus, accessible_gpus_for_seg, segments_per_host,
            dist_key_mapping, images_per_seg, custom_function_map=custom_function_map)

    loss = loss_metric[0]
    metric = loss_metric[1]

    if not metrics_type:
        metrics_type = None
        metric = None

    with MinWarning("error"):
        create_output_table = plpy.prepare("""
            CREATE TABLE {0} AS
            SELECT $1 as loss, $2 as metric, $3 as metrics_type, $4 as loss_type""".format(output_table), ["FLOAT", "FLOAT", "TEXT[]", "TEXT"])
        plpy.execute(create_output_table, [loss, metric, metrics_type, loss_type])

    if is_mult_model:
        plpy.execute("DROP VIEW IF EXISTS {0}".format(model_summary_table))

def validate_evaluate(module_name, model_table, model_summary_table, test_table, test_summary_table, output_table, is_mult_model):
    def _validate_test_summary_tbl():
        input_tbl_valid(test_summary_table, module_name,
                error_suffix_str="Please ensure that the test table ({0}) "
                                 "has been preprocessed by "
                                 "the image preprocessor.".format(test_table))
        cols_in_tbl_valid(test_summary_table, [NORMALIZING_CONST_COLNAME,
            DEPENDENT_VARTYPE_COLNAME, DEPENDENT_VARNAME_COLNAME,
            INDEPENDENT_VARNAME_COLNAME], module_name)

    input_tbl_valid(model_table, module_name)
    if is_mult_model and not columns_exist_in_table(model_table, ['mst_key']):
        plpy.error("{module_name}: Single model should not pass mst_key".format(**locals()))
    if not is_mult_model and columns_exist_in_table(model_table, ['mst_key']):
        plpy.error("{module_name}: Multi-model needs to pass mst_key".format(**locals()))
    InputValidator.validate_predict_evaluate_tables(
        module_name, model_table, model_summary_table,
        test_table, output_table)
    _validate_test_summary_tbl()

    dependent_varname = plpy.execute("SELECT {0} FROM {1}".format(
        "dependent_varname", model_summary_table))[0]["dependent_varname"]
    for i in dependent_varname:
        validate_bytea_var_for_minibatch(test_table, i)

def get_loss_metric_from_keras_eval(schema_madlib, table, dependent_varname,
                                    independent_varname, compile_params, fit_params,
                                    model_arch, serialized_weights, use_gpus,
                                    accessible_gpus_for_seg, segments_per_host,
                                    dist_key_mapping, images_per_seg,
                                    should_clear_session=True, custom_function_map=None,
                                    model_table=None, mst_key=None):
    """
    This function will call the internal keras evaluate function to get the loss
    and accuracy of each tuple which then gets averaged to get the final result.
    """

    dist_key_col = '0' if is_platform_pg() else '__table__.{0}'.format(DISTRIBUTION_KEY_COLNAME)
    gp_segment_id_col = '0' if is_platform_pg() else '__table__.{0}'.format(GP_SEGMENT_ID_COLNAME)

    """
    This function will call the internal keras evaluate function to get the loss
    and accuracy of each tuple which then gets averaged to get the final result.
    """
    use_gpus = use_gpus if use_gpus else False

    mb_dep_var_cols_sql = ', '.join(dependent_varname)
    mb_indep_var_cols_sql = ', '.join(independent_varname)
    dep_shape_cols = [add_postfix(i, "_shape") for i in dependent_varname]
    ind_shape_cols = [add_postfix(i, "_shape") for i in independent_varname]
    dep_shape_cols_sql = ', '.join(dep_shape_cols)
    ind_shape_cols_sql = ', '.join(ind_shape_cols)

    eval_sql = """
        select ({schema_madlib}.internal_keras_evaluate(
                                            ARRAY[{mb_dep_var_cols_sql}],
                                            ARRAY[{mb_indep_var_cols_sql}],
                                            ARRAY[{dep_shape_cols_sql}],
                                            ARRAY[{ind_shape_cols_sql}],
                                            $MAD${model_arch}$MAD$,
                                            {weights},
                                            {compile_params},
                                            {fit_params},
                                            {dist_key_col},
                                            ARRAY{dist_key_mapping},
                                            {gp_segment_id_col},
                                            ARRAY{segments_per_host},
                                            ARRAY{images_per_seg},
                                            ARRAY{accessible_gpus_for_seg},
                                            {should_clear_session},
                                            {custom_map_var}
                                            )) as loss_metric
        from {table} AS __table__ {mult_sql}
        """

    if mst_key:
        weights = '__mt__.{0}'.format(MODEL_WEIGHTS_COLNAME)
        mst_key_col = ModelSelectionSchema.MST_KEY
        mult_sql = ', {model_table} AS __mt__ WHERE {mst_key_col} = {mst_key}'.format(**locals())
        custom_map_var = '$1'
        evaluate_query = plpy.prepare(eval_sql.format(**locals()), ["bytea"])
        res = plpy.execute(evaluate_query, [custom_function_map])
    else:
        weights = '$1'
        mult_sql = ''
        custom_map_var = '$2'
        evaluate_query = plpy.prepare(eval_sql.format(**locals()), ["bytea", "bytea"])
        res = plpy.execute(evaluate_query, [serialized_weights, custom_function_map])


    if res is None:
        plpy.error("Zero rows returned from evaluate query: {}".format(evaluate_query))
    else:
        loss_metric = res[0]['loss_metric']
    return loss_metric

def internal_keras_eval_transition(state, dependent_var, independent_var,
                                   dependent_var_shape, independent_var_shape,
                                   model_architecture, serialized_weights, compile_params,fit_params,
                                   dist_key, dist_key_mapping, current_seg_id,
                                   segments_per_host, images_per_seg,
                                   accessible_gpus_for_seg, should_clear_session,
                                   custom_function_map=None, **kwargs):
    GD = kwargs['GD']
    device_name = get_device_name_and_set_cuda_env(accessible_gpus_for_seg[current_seg_id], current_seg_id)

    """
    This transition function is common to evaluate as well as the fit functions.
    All these calls have a different logic for creating and clear the tensorflow
    session
    For evaluate,
        We create only one tensorflow session and store it in GD.
        should_clear_session is always set to true, so the session and GD is
        cleared once the last buffer is evaluated on each segment.
    For fit,
        We reuse the session and GD created as part of fit_transition and only clear
        the session and GD at last row of the last iteration of eval_transition.
        should_clear_session is only set to true for the last call to eval_transition
        which can be either the training eval or validation eval
    For fit_multiple,
        We create one session per hop and store it in GD.
        should_clear_session is always set to true, so the session and GD is
        cleared once the last buffer is evaluated on each segment.
    """

    multi_output = True if len(dependent_var) > 1 else False

    record("seg {} enter eval_trans at {}\n".format(current_seg_id ,time.time()))

    if multi_output:
        output_count = len(dependent_var)
        agg_loss = state[0]
        if agg_loss == 0:
            state = []
            for i in range(2*output_count+2):
                state.append(0)
        agg_image_count = state[-1]
        aux_losses = []
        aux_metrics = []
        for counter in range(output_count):
            aux_losses.append(state[2*counter+1])
            aux_metrics.append(state[2*counter+2])

    else:
        agg_loss, agg_metric, agg_image_count = state

    segment_model, sess = get_init_model_and_sess(GD, device_name,
                                                  accessible_gpus_for_seg[current_seg_id],
                                                  segments_per_host[current_seg_id],
                                                  model_architecture,
                                                  compile_params, custom_function_map)
    record("seg {} restore eval model at {}\n".format(current_seg_id, time.time()))
    if not agg_image_count:
        # These should already be 0, but just in case make sure
        agg_metric = 0
        agg_loss = 0
        set_model_weights(segment_model, serialized_weights)

    x_val = []
    y_val = []
    for counter, shape in enumerate(independent_var_shape):
        x_val.append(np_array_float32(independent_var[counter], shape))
    for counter, shape in enumerate(dependent_var_shape):
        y_val.append(np_array_int16(dependent_var[counter], shape))

    image_count = len(y_val[0])
    agg_image_count += image_count

    X = [np.array(x_val[0][:,i]) for i in range(x_val[0].shape[1])]
    fit_params = parse_and_validate_fit_params(fit_params, current_seg_id)
    fit_params_new = {}
    if 'batch_size' in fit_params.keys():
        fit_params_new['batch_size'] =  fit_params['batch_size']

    f = open("/data2/ruike/tmp.txt", 'a')
    f.write(str(fit_params_new))
    f.write(str(fit_params))
    f.close()
    record("seg {} model eval at {}\n".format(current_seg_id, time.time()))
    res = segment_model.evaluate(X, y_val[0][:,0].reshape(-1, 1),  **fit_params_new)
    record("seg {} model finish eval at {}\n".format(current_seg_id, time.time()))
    # if metric is None, model.evaluate will only return loss as a scalar
    # Otherwise, it will return a list which has loss and metric
    if multi_output:
        loss = res[0]
        agg_loss += (image_count * loss)
        for counter in range(output_count):
            # For multi output cases, res has the following structure
            # print(model.metrics_names)
            # ['loss', 'dense_4_loss', 'dense_5_loss', 'dense_4_acc', 'dense_5_acc']
            aux_losses[counter] = aux_losses[counter] + (image_count * res[counter+1])
            aux_metrics[counter] = aux_metrics[counter] + (image_count * res[counter+1+len(dependent_var)])
    else:
        if type(res) is list:
           loss, metric = res
        else:
            loss = res
            metric = 0

        agg_loss += (image_count * loss)
        agg_metric += (image_count * metric)

    total_images = get_image_count_per_seg_from_array(dist_key_mapping.index(dist_key),
                                                      images_per_seg)
    is_last_row = agg_image_count == total_images
    if is_last_row and should_clear_session:
        GD_STORE.clear(GD)
        clear_keras_session(sess)
        del sess
        del segment_model

    state = [agg_loss]

    if multi_output:
        for counter in range(output_count):
            state.append(aux_losses[counter])
            state.append(aux_metrics[counter])
    else:
        state.append(agg_metric)
    state.append(agg_image_count)
    record("seg {} finish eval_trans at {}\n".format(current_seg_id ,time.time()))
    return state

def internal_keras_eval_merge(state1, state2, **kwargs):
    record("mdw enter eval_merge at {}\n".format(time.time()))
    # If either state is None, return the other one
    if not state1 or not state2:
        return state1 or state2

    merged_state = []
    for i in range(len(state1)):
        merged_state.append(state1[i]+state2[i])
    record("mdw finish eval_merge at {}\n".format(time.time()))
    return merged_state

def internal_keras_eval_final(state, **kwargs):
    image_count = state[-1]

    if image_count == 0:
        plpy.error("internal_keras_eval_final: Total images processed is 0")

    for i in range(len(state)-1):
        state[i] = state[i]/image_count

    return state

def fit_help(schema_madlib, message, **kwargs):
    """
    Help function for keras fit

    Args:
        @param schema_madlib
        @param message: string, Help message string
        @param kwargs

    Returns:
        String. Help/usage information
    """
    if not message:
        help_string = """
-----------------------------------------------------------------------
                            SUMMARY
-----------------------------------------------------------------------
This module allows you to use SQL to call deep learning
models designed in Keras, which is a high-level neural
network API written in Python.
Keras was developed for fast experimentation.  It can run
on top of different backends and the one that is currently
supported by MADlib is TensorFlow.  The implementation
in MADlib is distributed and designed to train
a single large model across multiple segments (workers)
in a Greenplum database.  PostgreSQL is also supported.

For more details on function usage:
    SELECT {schema_madlib}.madlib_keras_fit('usage')
            """
    elif message in ['usage', 'help', '?']:
        help_string = """
-----------------------------------------------------------------------
                            USAGE
-----------------------------------------------------------------------
 SELECT {schema_madlib}.madlib_keras_fit(
    source_table,               --  Name of the table containing the
                                    training data
    model,                      --  Name of the output table containing
                                    the model
    model_arch_table,           --  Name of the table containing the
                                    model architecture
    model_id,                   --  This is the id in 'model_arch_table'
                                    containing the model architecture
    compile_params,             --  Parameters passed to the compile
                                    method of the Keras model class
    fit_params,                 --  Parameters passed to the fit method
                                    of the Keras model class
    num_iterations,             --  Number of iterations to train.
    use_gpus,                   --  Flag to enable GPU support
    validation_table,           --  Name of the table containing
                                    the validation dataset
    metrics_compute_frequency,  --  Frequency to compute per-iteration
                                    metrics
    warm_start,                 --  Flag to enable warm start
    name,                       --  Free text string to identify a name
    description                 --  Free text string to provide a description
    )
 );

-----------------------------------------------------------------------
                            OUTPUT
-----------------------------------------------------------------------
The output table ('model' above) contains the following columns:

model_weights: Byte array containing the weights of the neural net.
model_arch: A JSON representation of the model architecture used in
            training.

A summary table ('<model>_summary') is created to store various training
statistics as well as the input parameters.
"""
    else:
        help_string = "No such option. Use {schema_madlib}.madlib_keras_fit()"

    return help_string.format(schema_madlib=schema_madlib)
# ---------------------------------------------------------------------


def evaluate_help(schema_madlib, message, **kwargs):
    """
    Help function for keras evaluate

    Args:
        @param schema_madlib
        @param message: string, Help message string
        @param kwargs

    Returns:
        String. Help/usage information
    """
    if not message:
        help_string = """
-----------------------------------------------------------------------
                            SUMMARY
-----------------------------------------------------------------------
This function allows the user to evaluate a madlib_keras_fit trained
model.

For more details on function usage:
    SELECT {schema_madlib}.madlib_keras_evaluate('usage')
            """
    elif message in ['usage', 'help', '?']:
        help_string = """
-----------------------------------------------------------------------
                            USAGE
-----------------------------------------------------------------------
 SELECT {schema_madlib}.madlib_keras_evaluate(
    model_table,    --  Name of the table containing the model
    test_table,     --  Name of the table containing the evaluation dataset
    output_table,   --  Name of the output table
    use_gpus,       --  Flag to enable GPU support
    mst_key         --  Identifier for the desired model out of multimodel
                        training output
    )
 );

-----------------------------------------------------------------------
                            OUTPUT
-----------------------------------------------------------------------
The output table ('output_table' above) contains the following columns:

loss:           Loss value on evaluation dataset.
metric:         Metric value on evaluation dataset, where 'metrics_type'
                below identifies the type of metric.
metrics_type:   Type of metric used that was used in the training step.
loss_type:      Type of loss used that was used in the training step.
"""
    else:
        help_string = "No such option. Use {schema_madlib}.madlib_keras_evaluate()"

    return help_string.format(schema_madlib=schema_madlib)
# ---------------------------------------------------------------------

@MinWarning("warning")
def fit_batch_level(schema_madlib, source_table, model, model_arch_table,
        model_id, compile_params, fit_params, samples_every, num_iterations,
        use_gpus, validation_table=None,
        metrics_compute_frequency=None, warm_start=False, name="",
        description="", object_table=None, **kwargs):

    import math
    module_name = 'madlib_keras_fit'
    fit_params = "" if not fit_params else fit_params
    _assert(compile_params, "Compile parameters cannot be empty or NULL.")

    input_tbl_valid(source_table, module_name)
    segments_per_host = get_data_distribution_per_segment(source_table)
    use_gpus = use_gpus if use_gpus else False
    if use_gpus:
        accessible_gpus_for_seg = get_accessible_gpus_for_seg(schema_madlib,
                                                              segments_per_host,
                                                              module_name)
    else:
        accessible_gpus_for_seg = get_seg_number()*[0]

    if object_table is not None:
        object_table = "{0}.{1}".format(schema_madlib, quote_ident(object_table))
    fit_validator = FitInputValidator(
        source_table, validation_table, model, model_arch_table, model_id,
        num_iterations, metrics_compute_frequency, warm_start,
        use_gpus, accessible_gpus_for_seg, object_table)

    multi_dep_count = len(fit_validator.dependent_varname)
    src_summary_dict = fit_validator.src_summary_dict
    class_values_colnames = [add_postfix(i, "_class_values") for i in
                             fit_validator.dependent_varname]

    if metrics_compute_frequency is None:
        metrics_compute_frequency = num_iterations

    warm_start = bool(warm_start)

    # The following two times must be recorded together.
    metrics_elapsed_start_time = time.time()
    start_training_time = datetime.datetime.now()
    #TODO add a unit test for this in a future PR
    # save the original value of the env variable so that we can reset it later.
    original_cuda_env = None
    if CUDA_VISIBLE_DEVICES_KEY in os.environ:
        original_cuda_env = os.environ[CUDA_VISIBLE_DEVICES_KEY]

    # Get the serialized master model
    start_deserialization = time.time()
    model_arch, model_weights = get_model_arch_weights(model_arch_table, model_id)

    # The last n layers are the output layers where n is the number of dep vars
    num_classes = get_num_classes(model_arch, multi_dep_count)

    input_shape = get_input_shape(model_arch)
    #fit_validator.validate_input_shapes(input_shape)

    dist_key_col = '0' if is_platform_pg() else DISTRIBUTION_KEY_COLNAME
    gp_segment_id_col = '0' if is_platform_pg() else GP_SEGMENT_ID_COLNAME
    embedding_weights_col = 'embedding_weights'
    embedding_bias_col = 'embedding_biass'
    serialized_weights = get_initial_weights(model, model_arch, model_weights,
                                             warm_start, accessible_gpus_for_seg)
    # Compute total images on each segment
    shape_col = fit_validator.dependent_shape_varname[0]
    dist_key_mapping, images_per_seg_train = \
        get_image_count_per_seg_for_minibatched_data_from_db(source_table,
                                                             shape_col)

    if validation_table:
        shape_col = fit_validator.val_dependent_shape_varname[0]
        dist_key_mapping_val, images_per_seg_val = \
            get_image_count_per_seg_for_minibatched_data_from_db(validation_table,
                                                                 shape_col)

    # Construct validation dataset if provided
    validation_set_provided = bool(validation_table)
    validation_metrics = []; validation_loss = []

    # Prepare the SQL for running distributed training via UDA
    compile_params_to_pass = quote_literal(compile_params)
    fit_params_to_pass = quote_literal(fit_params)
    custom_function_map = None

    # If the object_table exists, we read the list of custom
    # function used in the compile_params and map it to their
    # object definition from the object table
    custom_fn_list = get_custom_functions_list(compile_params)
    if object_table is not None:
        custom_function_map = query_custom_functions_map(object_table, custom_fn_list)
    elif len(custom_fn_list) >= 1:
        # Error out if custom_function is called without specifying the object table
        # with the function definition
        plpy.error("Object table not specified for function {0} in compile_params".format(custom_fn_list))

    # Use the smart interface
    if (len(fit_validator.dependent_varname) <= 5 and
            len(fit_validator.independent_varname) <= 5):

        dep_var_array = 5 * ["NULL"]
        indep_var_array = 5 * ["NULL"]

        for counter, var in enumerate(fit_validator.dependent_varname):
            dep_var_array[counter] = var

        for counter, var in enumerate(fit_validator.independent_varname):
            indep_var_array[counter] = var
        mb_dep_var_cols_sql = ', '.join(dep_var_array)
        mb_indep_var_cols_sql = ', '.join(indep_var_array)
    else:

        mb_dep_var_cols_sql = ', '.join(["dependent_var_{0}".format(i)
                                         for i in fit_validator.dependent_varname])
        mb_dep_var_cols_sql = "ARRAY[{0}]".format(mb_dep_var_cols_sql)

        mb_indep_var_cols_sql = ', '.join(["independent_var_{0}".format(i)
                                           for i in fit_validator.independent_varname])
        mb_indep_var_cols_sql = "ARRAY[{0}]".format(mb_indep_var_cols_sql)

    dep_shape_cols_sql = ', '.join(fit_validator.dependent_shape_varname)
    ind_shape_cols_sql = ', '.join(fit_validator.independent_shape_varname)



    # Define the state for the model and loss/metric storage lists
    training_loss, training_metrics, metrics_elapsed_time = [], [], []
    metrics_iters = []

    # get the size of serialized model weights string in KB
    model_size = sys.getsizeof(serialized_weights)/1024.0
    record("model_size: {} \n".format(model_size ))
    buffer_size = get_buffer_size(source_table,  shape_col)
    samples_every = int(round(samples_every / buffer_size) * buffer_size)
    slot_every = round(samples_every / buffer_size)
    plpy.info("\n"+ "Use samples_every: {}".format(samples_every ))
    # Run distributed training for specified number of iterations
    for i in range(1, num_iterations+1):
        record("begin {} iteration at {}\n".format(i ,time.time()))
        start_iteration = time.time()
        is_final_iteration = (i == num_iterations)
        total_images = get_image_count_per_seg_from_array(0, images_per_seg_train)
        inner_iteration = math.ceil(total_images / float(samples_every))
        plpy.info("\n"+ "total_images: {} Total inner_iteration: {}".format(total_images, inner_iteration))
        start_slot = 1
        end_slot = 1 + slot_every
        for j in range(1, int(inner_iteration)):
            try:
                target_buffer = list()
                target_buffer.append((start_slot - 1)*4)
                target_buffer.append((start_slot - 1)*4 + 1)
                target_buffer.append((start_slot - 1)*4 + 2)
                target_buffer.append((start_slot - 1)*4 + 3)
                #https://stackoverflow.com/questions/30896497/postgres-column-does-not-exist-but-its-there-with-alias
                run_training_iteration = plpy.prepare("""
        ;WITH buffer_ids AS (
            SELECT unnest(array{target_buffer}) AS buffer_id
        ),
        subqueries AS (
            SELECT
                buffer_id,
                array_agg(embedding_weight ORDER BY t.byte) AS embedding_weights,
                array_agg(embedding_bias ORDER BY t.byte) AS embedding_biass
            FROM (
                SELECT DISTINCT buffer_id, get_byte(xi, generate_series(0, octet_length(xi)-1)) AS byte
                FROM {source_table}
                WHERE buffer_id IN (SELECT buffer_id FROM buffer_ids)
                ) t
            LEFT JOIN embed_model e ON t.byte = e.id
            GROUP BY buffer_id
        ),
        TMP_TABLE AS (
            SELECT {source_table}.*, subqueries.embedding_weights, subqueries.embedding_biass,{source_table}.gp_segment_id
            FROM {source_table}
            JOIN subqueries ON {source_table}.buffer_id = subqueries.buffer_id
        )
        SELECT {schema_madlib}.fit_step_batch_level(
            {samples_every},
            {mb_dep_var_cols_sql},
            {mb_indep_var_cols_sql},
            {embedding_weights_col},
            {embedding_bias_col},
            ARRAY[{dep_shape_cols_sql}],
            ARRAY[{ind_shape_cols_sql}],
            $MAD${model_arch}$MAD$::TEXT,
            {compile_params_to_pass}::TEXT,
            {fit_params_to_pass}::TEXT,
            {dist_key_col},
            ARRAY{dist_key_mapping},
            {gp_segment_id_col},
            ARRAY{segments_per_host},
            ARRAY{images_per_seg_train},
            ARRAY{accessible_gpus_for_seg},
            $1,
            $2
        ) AS iteration_result
        FROM TMP_TABLE
        """.format(**locals()), ["bytea", "bytea"])
                record("mdw send model at {}\n".format(time.time()))
                serialized_weights = plpy.execute(run_training_iteration,
                                              [serialized_weights, custom_function_map]
                                              )[0]['iteration_result']
                record("mdw finish fit_step at {}\n".format(time.time()))
                model_size = sys.getsizeof(serialized_weights)/1024.0
                record("model_size: {} \n".format(model_size ))
                start_slot = start_slot + slot_every
                end_slot = end_slot + slot_every
            except plpy.SPIError as e:
                msg = e.message
                if 'TransAggDetail' in msg:
                    e.message, detail = msg.split('TransAggDetail')
                elif 'MergeAggDetail' in msg:
                    e.message, detail = msg.split('MergeAggDetail')
                elif 'FinalAggDetail' in msg:
                    e.message, detail = msg.split('FinalAggDetail')
                else:
                    raise e
            # Extract Traceback from segment, add to
            #  DETAIL of error message on coordinator
                e.args = (e.message,)
                spidata = list(e.spidata)
                spidata[1] = detail
                e.spidata = tuple(spidata)
                raise e

        end_iteration = time.time()
        info_str = "\tTime for training in iteration {0}: {1} sec".format(i,
                                                                          end_iteration - start_iteration)

        if should_compute_metrics_this_iter(i, metrics_compute_frequency,
                                            num_iterations):
            """
            If there is no validation dataset, we should clear the session/gd at
            the last call to train evaluate. Otherwise clear it at the last call
            to validation evaluate
            """
            should_clear_session = False
            if not validation_set_provided:
                should_clear_session = is_final_iteration

            record("mdw begin compute_loss_and_metrics for train set at {}\n".format(time.time()))
            compute_out = compute_loss_and_metrics(schema_madlib, source_table,
                                                   fit_validator.dependent_varname,
                                                   fit_validator.independent_varname,
                                                   compile_params_to_pass,
                                                   fit_params_to_pass,
                                                   model_arch,
                                                   serialized_weights, use_gpus,
                                                   accessible_gpus_for_seg,
                                                   segments_per_host,
                                                   dist_key_mapping,
                                                   images_per_seg_train,
                                                   training_metrics,
                                                   training_loss,
                                                   should_clear_session,
                                                   custom_function_map)
            record("mdw finish compute_loss_and_metrics for train set at {}\n".format(time.time()))
            metrics_iters.append(i)
            compute_time, compute_metrics, compute_loss = compute_out
            info_str = get_evaluate_info_msg(i, info_str, compute_out, True)
            if validation_set_provided:
                # Compute loss/accuracy for validation data.
                record("mdw begin compute_loss_and_metrics for vaild set at {}\n".format(time.time()))
                val_compute_out = compute_loss_and_metrics(schema_madlib,
                                                           validation_table,
                                                           fit_validator.val_dependent_varname,
                                                           fit_validator.val_independent_varname,
                                                           compile_params_to_pass,
                                                           fit_params_to_pass,
                                                           model_arch,
                                                           serialized_weights,
                                                           use_gpus,
                                                           accessible_gpus_for_seg,
                                                           segments_per_host,
                                                           dist_key_mapping_val,
                                                           images_per_seg_val,
                                                           validation_metrics,
                                                           validation_loss,
                                                           is_final_iteration,
                                                           custom_function_map)
                info_str = get_evaluate_info_msg(i, info_str, val_compute_out,
                                                 False)
                record("mdw finish compute_loss_and_metrics for vaild set at {}\n".format(time.time()))

            metrics_elapsed_end_time = time.time()
            metrics_elapsed_time.append(
                metrics_elapsed_end_time-metrics_elapsed_start_time)
        plpy.info("\n"+info_str)
        record("finish {} iteration at {}\n".format(i ,time.time()))

    end_training_time = datetime.datetime.now()

    version = madlib_version(schema_madlib)
    norm_const = src_summary_dict['normalizing_const']
    dep_vartype = src_summary_dict['dependent_vartype']
    dependent_varname = src_summary_dict['dependent_varname']
    independent_varname = src_summary_dict['independent_varname']

    dep_name_list = ', '.join([quote_literal(i) for i in dependent_varname])
    ind_name_list = ', '.join([quote_literal(i) for i in independent_varname])

    # Define some constants to be inserted into the summary table.
    model_type = "madlib_keras"
    metrics_list = get_metrics_from_compile_param(compile_params)
    is_metrics_specified = True if metrics_list else False
    metrics_type = 'ARRAY{0}'.format(metrics_list) if is_metrics_specified else 'NULL'
    metrics_iters = metrics_iters if metrics_iters else 'NULL'
    loss_type = get_loss_from_compile_param(compile_params)

    # We always compute the training loss and metrics, at least once.
    training_metrics_final, training_metrics = get_metrics_sql_string(
        training_metrics, is_metrics_specified)
    training_loss_final, training_loss = get_metrics_sql_string(
        training_loss, True)

    # Validation loss and metrics are computed only if validation_table
    # is provided.
    if validation_set_provided:
        validation_metrics_final, validation_metrics = get_metrics_sql_string(
            validation_metrics, is_metrics_specified)
        validation_loss_final, validation_loss = get_metrics_sql_string(validation_loss)
        # Must quote the string before inserting to table. Explicitly
        # quoting it here since this can also take a NULL value, done
        # in the else part.
        validation_table = quote_literal(validation_table)
    else:
        validation_metrics = validation_loss = 'NULL'
        validation_metrics_final = validation_loss_final = 'NULL'
        validation_table = 'NULL'

    object_table = quote_literal(object_table) if object_table is not None else 'NULL'
    class_values_colnames = ' , '.join(class_values_colnames)
    if warm_start:
        plpy.execute("DROP TABLE {0}, {1}".format
                     (model, fit_validator.output_summary_model_table))
    create_output_summary_table = plpy.prepare("""
        CREATE TABLE {output_summary_model_table} AS
        SELECT
            $MAD${source_table}$MAD$::TEXT AS source_table,
            $MAD${model}$MAD$::TEXT AS model,
            ARRAY[{dep_name_list}]::TEXT[] AS dependent_varname,
            ARRAY[{ind_name_list}]::TEXT[] AS independent_varname,
            $MAD${model_arch_table}$MAD$::TEXT AS model_arch_table,
            {model_id}::INTEGER AS {model_id_colname},
            $1 AS compile_params,
            $2 AS fit_params,
            {num_iterations}::INTEGER AS num_iterations,
            {validation_table}::TEXT AS validation_table,
            {object_table}::TEXT AS object_table,
            {metrics_compute_frequency}::INTEGER AS metrics_compute_frequency,
            $3 AS name,
            $4 AS description,
            '{model_type}'::TEXT AS model_type,
            {model_size}::DOUBLE PRECISION AS model_size,
            '{start_training_time}'::TIMESTAMP AS start_training_time,
            '{end_training_time}'::TIMESTAMP AS end_training_time,
            $5 AS metrics_elapsed_time,
            '{version}'::TEXT AS madlib_version,
            ARRAY{num_classes}::INTEGER[] AS num_classes,
            ARRAY{dep_vartype}::TEXT[] AS {dependent_vartype_colname},
            {norm_const}::{FLOAT32_SQL_TYPE} AS {normalizing_const_colname},
            {metrics_type}::TEXT[] AS metrics_type,
            '{loss_type}'::TEXT AS loss_type,
            {training_metrics_final}::DOUBLE PRECISION AS training_metrics_final,
            {training_loss_final}::DOUBLE PRECISION AS training_loss_final,
            {training_metrics}::DOUBLE PRECISION[] AS training_metrics,
            {training_loss}::DOUBLE PRECISION[] AS training_loss,
            {validation_metrics_final}::DOUBLE PRECISION AS validation_metrics_final,
            {validation_loss_final}::DOUBLE PRECISION AS validation_loss_final,
            {validation_metrics}::DOUBLE PRECISION[] AS validation_metrics,
            {validation_loss}::DOUBLE PRECISION[] AS validation_loss,
            ARRAY{metrics_iters}::INTEGER[] AS metrics_iters,
            {class_values_colnames}
        FROM {source_summary_table}
        """.format(output_summary_model_table=fit_validator.output_summary_model_table,
                   dependent_vartype_colname=DEPENDENT_VARTYPE_COLNAME,
                   normalizing_const_colname=NORMALIZING_CONST_COLNAME,
                   FLOAT32_SQL_TYPE = FLOAT32_SQL_TYPE,
                   model_id_colname = ModelArchSchema.MODEL_ID,
                   source_summary_table=fit_validator.source_summary_table,
                   **locals()),
                                               ["TEXT", "TEXT", "TEXT", "TEXT", "DOUBLE PRECISION[]"])
    plpy.execute(create_output_summary_table,
                 [compile_params, fit_params, name,
                  description, metrics_elapsed_time])

    plpy.execute("""
        CREATE TABLE {0}
        (model_weights bytea,
        {1} json)""".format(model, ModelArchSchema.MODEL_ARCH))
    insert_output_table = plpy.prepare("""
        INSERT INTO {0} SELECT model_weights, {1}
        FROM (VALUES($1, $2))t(model_weights, {1})
        """.format(model, ModelArchSchema.MODEL_ARCH), ["bytea", "json"])
    plpy.execute(insert_output_table, [serialized_weights, model_arch])

    #TODO add a unit test for this in a future PR
    reset_cuda_env(original_cuda_env)


def fit_transition_batch_level(state, samples_every, dependent_var, independent_var, embedding_weights, dependent_var_shape,
                   independent_var_shape, model_architecture,
                   compile_params, fit_params, dist_key, dist_key_mapping,
                   current_seg_id, segments_per_host, images_per_seg,
                   accessible_gpus_for_seg, prev_serialized_weights,
                   is_multiple_model=False, custom_function_map=None, **kwargs):
    """
    This transition function is common for madlib_keras_fit() and
    madlib_keras_fit_multiple_model(). The important difference between
    these two calls is the way tensorflow/keras sessions and GD gets used.
    For madlib_keras_fit_multiple_model,
        a. We create a tensorflow session per hop and store it in GD alongwith
        the model and clear both GD and the session at the end of each
        hop.
    For madlib_keras_fit,
        b. We create only one tensorflow session for both fit and eval transition
        functions and store it in GD. This session gets reused by both fit and eval
        and only gets cleared in eval transition at the last row of the last iteration.

    """
    if not dependent_var_shape[0] or not independent_var_shape[0] \
            or dependent_var[0] is None or independent_var[0] is None:
        plpy.error("fit_transition called with no data")

    if not prev_serialized_weights or not model_architecture:
        return state

    current_seg_id = dist_key_mapping.index(dist_key)
    record("seg {} enter fit_trans at {}\n".format(current_seg_id ,time.time()))
    GD = kwargs['GD']

    trans_enter_time = time.time()

    record("embedding_Weights : {}".format(embedding_weights))
    record("independent_var_shape : {}".format(independent_var_shape))

    device_name = get_device_name_and_set_cuda_env(accessible_gpus_for_seg[current_seg_id], current_seg_id)

    '''segment_model, sess = get_init_model_and_sess(GD, device_name,
                                                  accessible_gpus_for_seg[current_seg_id],
                                                  segments_per_host[current_seg_id],
                                                  model_architecture, compile_params,
                                                  custom_function_map)

    if GD_STORE.AGG_IMAGE_COUNT in GD:
        record("seg {} restore model at {}\n".format(current_seg_id, time.time()))
        agg_image_count = GD[GD_STORE.AGG_IMAGE_COUNT]
    else:
        record("seg {} receive model at {}\n".format(current_seg_id, time.time()))
        agg_image_count = 0
        GD[GD_STORE.AGG_IMAGE_COUNT] = agg_image_count
        set_model_weights(segment_model, prev_serialized_weights)
    record("seg {} finish set model at {}\n".format(current_seg_id, time.time()))'''

    x_train = []
    y_train = []
    # Prepare the data
    for counter, shape in enumerate(independent_var_shape):
        x_train.append(np_array_float32(independent_var[counter], shape))

    for counter, shape in enumerate(dependent_var_shape):
        y_train.append(np_array_int16(dependent_var[counter], shape))

    # Fit segment model on data
    #TODO consider not doing this every time
    fit_params = parse_and_validate_fit_params(fit_params, current_seg_id)
    x_train_tmp = x_train[0].T
    X = [np.array(x_train_tmp[i, :]) for i in range(x_train_tmp.shape[0])]
    record("x_train : {}".format(x_train))
    embed_id_unique = np.unique(np.array(X))
    embedding = list()
    #embedding_bias = list()
    embedding_mapping = dict()
    for i in range(len(embedding_weights)):
        embedding.append(tf_deserialize_embedding(embedding_weights[i]))
        #embedding_bias.append(tf_deserialize_embedding(embedding_bias_weights[i]))
        embedding_mapping[embed_id_unique[i]] = i
    record("embed_id_unique : {}".format(embed_id_unique))
    record("embedding : {}".format(embedding))
    record("seg {} train model at {}\n".format(current_seg_id, time.time()))
    segment_model.fit(X, y_train[0][:,0].reshape(-1, 1), **fit_params)
    # Aggregating number of images, loss and accuracy
    record("seg {} finish train model at {}\n".format(current_seg_id, time.time()))

    agg_image_count += dependent_var_shape[0][0]
    GD[GD_STORE.AGG_IMAGE_COUNT] = agg_image_count
    total_images = get_image_count_per_seg_from_array(dist_key_mapping.index(dist_key),
                                                      images_per_seg)
    is_last_row = agg_image_count > (samples_every - dependent_var_shape[0][0])
    record("seg {}: agg_image_count:{}, samples_every:{}\n".format(current_seg_id, agg_image_count, samples_every))
    return_state = get_state_to_return(segment_model, is_last_row, is_multiple_model,
                                       agg_image_count, agg_image_count)

    insert_output_table = plpy.prepare("""
        INSERT INTO gradients (id, grads) VALUES ({}, 1)
        """.format(current_seg_id))
    plpy.execute(insert_output_table)

    if is_last_row:
        del GD[GD_STORE.AGG_IMAGE_COUNT]  # Must be reset after each pass through images
        if is_multiple_model:
            GD_STORE.clear(GD)
            clear_keras_session(sess)

    trans_exit_time = time.time()
    DEBUG.plpy.info("|_fit_transition_time_|{}|".format(trans_exit_time - trans_enter_time))
    record("seg {} leave fit_trans at {}\n".format(current_seg_id ,time.time()))
    record("seg {} fit_trans last {}\n".format(current_seg_id ,trans_exit_time - trans_enter_time))
    return return_state

def fit_transition_wide_batch_level(state, samples_every, dependent_var1, dependent_var2, dependent_var3,
                   dependent_var4, dependent_var5,embedding_weights, independent_var1,
                   independent_var2, independent_var3, independent_var4,
                   independent_var5, dependent_var_shape,
                   independent_var_shape, model_architecture,
                   compile_params, fit_params, dist_key, dist_key_mapping,
                   current_seg_id, segments_per_host, images_per_seg,
                   accessible_gpus_for_seg, prev_serialized_weights,
                   is_multiple_model=False, custom_function_map=None, **kwargs):

    if not independent_var1 or not dependent_var1:
        return state
    dependent_var = [dependent_var1, dependent_var2, dependent_var3,
                        dependent_var4, dependent_var5]
    independent_var = [independent_var1, independent_var2, independent_var3,
                        independent_var4, independent_var5]

    dependent_var = [i for i in dependent_var if i is not None]
    independent_var = [i for i in independent_var if i is not None]

    return fit_transition_batch_level(state, samples_every, dependent_var, independent_var, embedding_weights,dependent_var_shape,
                   independent_var_shape, model_architecture,
                   compile_params, fit_params, dist_key, dist_key_mapping,
                   current_seg_id, segments_per_host, images_per_seg,
                   accessible_gpus_for_seg, prev_serialized_weights,
                   is_multiple_model, custom_function_map, **kwargs)


import os
import pdb
import sys
import numpy as np
import psycopg2 as p2
import threading
import tensorflow as tf
import datetime
import time
from collections import OrderedDict
from decimal import *

from sklearn.metrics import classification_report
from tensorflow import float32

from tf_utils import tf_serialize_nd_weights,  tf_deserialize_as_nd_weights,  tf_deserialize_gradient,  tf_serialize_gradient, tf_serialize_embedding, tf_deserialize_embedding, tf_deserialize_1d_weights, tf_serialize_1d_weights
from tensorflow.python.framework.ops import IndexedSlicesValue
sys.path.append('..')
from DeepFM import DeepFM,DeepFM_DA,DeepFM_Master,DeepFM_Worker
import config

#logfile = os.path.join('../logs', 'da_' + str(int(time.time())) + '.res')
logfile = './log'

def log_record(content, ifprint = True):
    with open(logfile, 'a') as f:
        ct = str(datetime.datetime.now())
        content_out = '[' + ct + '] ' + '[DA] ' + str(content)
        f.write(str(content_out) + '\n')
        if ifprint:
            print(content_out)

class Schema:
    Embed_Model_Table = 'embed_model'
    Dense_Model_Table = 'dense_model'
    Embed_GRADIENT_TABLE = 'embed_gradient_table'
    Dense_GRADIENT_TABLE = 'dense_gradient_table'
    worker = 'model_worker'
    master = 'model_master'

class tf_GD_STORE:
    SEGMENT_MODEL = 'model_worker'
    AGG_IMAGE_COUNT = 'agg_image_count'

    @staticmethod
    def init(tf_GD, segment_model):
        tf_GD[tf_GD_STORE.SEGMENT_MODEL] = segment_model

    @staticmethod
    def clear(tf_GD):
        del tf_GD[tf_GD_STORE.SEGMENT_MODEL]
        if tf_GD_STORE.AGG_IMAGE_COUNT in tf_GD:
            del tf_GD[tf_GD_STORE.AGG_IMAGE_COUNT]

def tf_get_init_model_and_sess(GD, current_seg_id, master_id):
    if GD_STORE.SEGMENT_MODEL in GD:
        # If a live session is present, re-use it. Otherwise, recreate it.
        segment_model = GD[GD_STORE.SEGMENT_MODEL]
        record("seg {} restored_model".format(current_seg_id))
    else:
        if current_seg_id == master_id:
            model_worker = DeepFM_Worker(current_seg_id, **dfm_params)
            GD_STORE.init(GD,model_worker)
            segment_model = GD[GD_STORE.SEGMENT_MODEL]
            record("seg {} create model".format(current_seg_id))
        else:
            model_worker = DeepFM_Worker(current_seg_id, **dfm_params)
            GD_STORE.init(GD,model_worker)
            segment_model = GD[GD_STORE.SEGMENT_MODEL]
            record("seg {} create model".format(current_seg_id))
    return segment_model

dfm_params = {
        "use_fm": True,
        "use_deep": True,
        "embedding_size": 128,
        "dropout_fm": [1.0, 1.0],
        "deep_layers": [128, 128],
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
        "random_seed": config.RANDOM_SEED
    }

def get_unique_id(xi, shape, **kwargs):
    xv=list()
    xv.append(np_array_float32(xi, shape))
    unique_id = np.unique(np.array(xv,dtype=np.int16))
    return unique_id

def model_test(xi, shape, **kwargs):
    xv=list()
    xv.append(np_array_float32(xi, shape))
    unique_id = np.unique(np.array(xv,dtype=np.int16))
    return unique_id

@MinWarning("warning")
def fit_gradients_batch_level(schema_madlib, source_table, model, model_arch_table,
        model_id, compile_params, fit_params, samples_every, num_iterations,
        use_gpus, validation_table=None,
        metrics_compute_frequency=None, warm_start=False, name="",
        description="", object_table=None, **kwargs):

    import math
    module_name = 'madlib_keras_fit'
    fit_params = "" if not fit_params else fit_params
    _assert(compile_params, "Compile parameters cannot be empty or NULL.")

    input_tbl_valid(source_table, module_name)
    segments_per_host = get_data_distribution_per_segment(source_table)
    use_gpus = use_gpus if use_gpus else False
    if use_gpus:
        accessible_gpus_for_seg = get_accessible_gpus_for_seg(schema_madlib,
                                                              segments_per_host,
                                                              module_name)
    else:
        accessible_gpus_for_seg = get_seg_number()*[0]

    if object_table is not None:
        object_table = "{0}.{1}".format(schema_madlib, quote_ident(object_table))
    fit_validator = FitInputValidator(
        source_table, validation_table, model, model_arch_table, model_id,
        num_iterations, metrics_compute_frequency, warm_start,
        use_gpus, accessible_gpus_for_seg, object_table)

    multi_dep_count = len(fit_validator.dependent_varname)
    src_summary_dict = fit_validator.src_summary_dict
    class_values_colnames = [add_postfix(i, "_class_values") for i in
                             fit_validator.dependent_varname]

    if metrics_compute_frequency is None:
        metrics_compute_frequency = num_iterations

    warm_start = bool(warm_start)

    # The following two times must be recorded together.
    metrics_elapsed_start_time = time.time()
    start_training_time = datetime.datetime.now()
    #TODO add a unit test for this in a future PR
    # save the original value of the env variable so that we can reset it later.
    original_cuda_env = None
    if CUDA_VISIBLE_DEVICES_KEY in os.environ:
        original_cuda_env = os.environ[CUDA_VISIBLE_DEVICES_KEY]

    # Get the serialized master model
    start_deserialization = time.time()

    # The last n layers are the output layers where n is the number of dep vars
    num_classes = 2

    input_shape = 2
    #fit_validator.validate_input_shapes(input_shape)

    dist_key_col = '0' if is_platform_pg() else DISTRIBUTION_KEY_COLNAME
    gp_segment_id_col = '0' if is_platform_pg() else GP_SEGMENT_ID_COLNAME

    # Compute total images on each segment
    shape_col = fit_validator.dependent_shape_varname[0]
    dist_key_mapping, images_per_seg_train = \
        get_image_count_per_seg_for_minibatched_data_from_db(source_table,
                                                             shape_col)

    if validation_table:
        shape_col = fit_validator.val_dependent_shape_varname[0]
        dist_key_mapping_val, images_per_seg_val = \
            get_image_count_per_seg_for_minibatched_data_from_db(validation_table,
                                                                 shape_col)

    # Construct validation dataset if provided
    validation_set_provided = bool(validation_table)
    validation_metrics = []; validation_loss = []

    # Prepare the SQL for running distributed training via UDA
    compile_params_to_pass = quote_literal(compile_params)
    fit_params_to_pass = quote_literal(fit_params)
    custom_function_map = None

    # If the object_table exists, we read the list of custom
    # function used in the compile_params and map it to their
    # object definition from the object table
    custom_fn_list = get_custom_functions_list(compile_params)
    if object_table is not None:
        custom_function_map = query_custom_functions_map(object_table, custom_fn_list)
    elif len(custom_fn_list) >= 1:
        # Error out if custom_function is called without specifying the object table
        # with the function definition
        plpy.error("Object table not specified for function {0} in compile_params".format(custom_fn_list))

    # Use the smart interface
    if (len(fit_validator.dependent_varname) <= 5 and
            len(fit_validator.independent_varname) <= 5):

        dep_var_array = 5 * ["NULL"]
        indep_var_array = 5 * ["NULL"]

        for counter, var in enumerate(fit_validator.dependent_varname):
            dep_var_array[counter] = var

        for counter, var in enumerate(fit_validator.independent_varname):
            indep_var_array[counter] = var
        mb_dep_var_cols_sql = ', '.join(dep_var_array)
        mb_indep_var_cols_sql = ', '.join(indep_var_array)
    else:

        mb_dep_var_cols_sql = ', '.join(["dependent_var_{0}".format(i)
                                         for i in fit_validator.dependent_varname])
        mb_dep_var_cols_sql = "ARRAY[{0}]".format(mb_dep_var_cols_sql)

        mb_indep_var_cols_sql = ', '.join(["independent_var_{0}".format(i)
                                           for i in fit_validator.independent_varname])
        mb_indep_var_cols_sql = "ARRAY[{0}]".format(mb_indep_var_cols_sql)

    dep_shape_cols_sql = ', '.join(fit_validator.dependent_shape_varname)
    ind_shape_cols_sql = ', '.join(fit_validator.independent_shape_varname)

    embedding_weights_col = 'embedding_weights'
    embedding_bias_col = 'embedding_w_bias'

    # Define the state for the model and loss/metric storage lists
    training_loss, training_metrics, metrics_elapsed_time = [], [], []
    metrics_iters = []

    model_master = DeepFM_Master(**dfm_params)
    del model_master
    # get the size of serialized model weights string in KB
    serialized_weights = [np.array([0])]
    serialized_weights = np.float32(serialized_weights).tostring()
    buffer_size = get_buffer_size(source_table,  shape_col)
    samples_every = int(round(samples_every / buffer_size) * buffer_size)
    slot_every = round(samples_every / buffer_size)
    plpy.info("\n"+ "Use samples_every: {}".format(samples_every))
    # Run distributed training for specified number of iterations
    plpy.info("\n" + "dist_key_map : {}\n".format(dist_key_mapping))
    for i in range(1, num_iterations+1):
        record("begin {} iteration at {}\n".format(i ,time.time()))
        start_iteration = time.time()
        is_final_iteration = (i == num_iterations)
        total_images = get_image_count_per_seg_from_array(0, images_per_seg_train)
        inner_iteration = math.ceil(total_images / float(samples_every))
        plpy.info("\n"+ "total_images: {} Total inner_iteration: {}".format(total_images, inner_iteration))
        start_slot = 1
        end_slot = 1 + slot_every
        for j in range(1, int(inner_iteration)+1):
            try:
                #https://stackoverflow.com/questions/30896497/postgres-column-does-not-exist-but-its-there-with-alias
                run_training_iteration = plpy.prepare("""
        ;WITH TMP_TABLE AS(
            SELECT *, gp_segment_id, ROW_NUMBER() OVER( PARTITION BY __dist_key__ ) slot_id
            FROM {source_table}
        )
        SELECT {schema_madlib}.fit_step_gradients_batch_level(
            {samples_every},
            {mb_dep_var_cols_sql},
            {mb_indep_var_cols_sql},
            ARRAY[{dep_shape_cols_sql}],
            ARRAY[{ind_shape_cols_sql}],
            {compile_params_to_pass}::TEXT,
            {fit_params_to_pass}::TEXT,
            {dist_key_col},
            ARRAY{dist_key_mapping},
            {gp_segment_id_col},
            ARRAY{segments_per_host},
            ARRAY{images_per_seg_train},
            ARRAY{accessible_gpus_for_seg},
            $1,
            $2
        ) AS iteration_result
        FROM TMP_TABLE
        WHERE slot_id >= {start_slot}
            AND slot_id < {end_slot}
        """.format(**locals()), ["bytea", "bytea"])
                record("mdw send model at {}\n".format(time.time()))
                serialized_weights = plpy.execute(run_training_iteration,
                                              [serialized_weights, custom_function_map]
                                              )[0]['iteration_result']
                record("mdw finish fit_step at {}\n".format(time.time()))
                if j == int(inner_iteration):
                    model_master.save_dense_weight(serialized_weights)
                model_master.apply_embedding_grads_udaf()
                t1 = time.time()
                start_slot = start_slot + slot_every
                end_slot = end_slot + slot_every
            except plpy.SPIError as e:
                msg = e.message
                if 'TransAggDetail' in msg:
                    e.message, detail = msg.split('TransAggDetail')
                elif 'MergeAggDetail' in msg:
                    e.message, detail = msg.split('MergeAggDetail')
                elif 'FinalAggDetail' in msg:
                    e.message, detail = msg.split('FinalAggDetail')
                else:
                    raise e
            # Extract Traceback from segment, add to
            #  DETAIL of error message on coordinator
                e.args = (e.message,)
                spidata = list(e.spidata)
                spidata[1] = detail
                e.spidata = tuple(spidata)
                raise e

        end_iteration = time.time()
        info_str = "\tTime for training in iteration {0}: {1} sec".format(i,
                                                                          end_iteration - start_iteration)

        if should_compute_metrics_this_iter(i, metrics_compute_frequency,
                                            num_iterations):
            """
            If there is no validation dataset, we should clear the session/gd at
            the last call to train evaluate. Otherwise clear it at the last call
            to validation evaluate
            """
            should_clear_session = False
            if not validation_set_provided:
                should_clear_session = is_final_iteration

            '''record("mdw begin compute_loss_and_metrics for train set at {}\n".format(time.time()))
            compute_out = compute_loss_and_metrics(schema_madlib, source_table,
                                                   fit_validator.dependent_varname,
                                                   fit_validator.independent_varname,
                                                   compile_params_to_pass,
                                                   fit_params_to_pass,
                                                   model_arch,
                                                   serialized_weights, use_gpus,
                                                   accessible_gpus_for_seg,
                                                   segments_per_host,
                                                   dist_key_mapping,
                                                   images_per_seg_train,
                                                   training_metrics,
                                                   training_loss,
                                                   should_clear_session,
                                                   custom_function_map)
            record("mdw finish compute_loss_and_metrics for train set at {}\n".format(time.time()))
            metrics_iters.append(i)
            compute_time, compute_metrics, compute_loss = compute_out
            info_str = get_evaluate_info_msg(i, info_str, compute_out, True)'''
            if validation_set_provided:
                # Compute loss/accuracy for validation data.
                record("mdw begin compute_loss_and_metrics for vaild set at {}\n".format(time.time()))
                val_compute_out = compute_loss_and_metrics(schema_madlib,
                                                           validation_table,
                                                           fit_validator.val_dependent_varname,
                                                           fit_validator.val_independent_varname,
                                                           compile_params_to_pass,
                                                           fit_params_to_pass,
                                                           serialized_weights,
                                                           use_gpus,
                                                           accessible_gpus_for_seg,
                                                           segments_per_host,
                                                           dist_key_mapping_val,
                                                           images_per_seg_val,
                                                           validation_metrics,
                                                           validation_loss,
                                                           is_final_iteration,
                                                           custom_function_map)
                info_str = get_evaluate_info_msg(i, info_str, val_compute_out,
                                                 False)
                record("mdw finish compute_loss_and_metrics for vaild set at {}\n".format(time.time()))

            metrics_elapsed_end_time = time.time()
            metrics_elapsed_time.append(
                metrics_elapsed_end_time-metrics_elapsed_start_time)
        plpy.info("\n"+info_str)
        record("finish {} iteration at {}\n".format(i ,time.time()))

    end_training_time = datetime.datetime.now()

    version = madlib_version(schema_madlib)
    norm_const = src_summary_dict['normalizing_const']
    dep_vartype = src_summary_dict['dependent_vartype']
    dependent_varname = src_summary_dict['dependent_varname']
    independent_varname = src_summary_dict['independent_varname']

    dep_name_list = ', '.join([quote_literal(i) for i in dependent_varname])
    ind_name_list = ', '.join([quote_literal(i) for i in independent_varname])

    # Define some constants to be inserted into the summary table.
    model_type = "madlib_keras"
    metrics_list = get_metrics_from_compile_param(compile_params)
    is_metrics_specified = True if metrics_list else False
    metrics_type = 'ARRAY{0}'.format(metrics_list) if is_metrics_specified else 'NULL'
    metrics_iters = metrics_iters if metrics_iters else 'NULL'
    loss_type = get_loss_from_compile_param(compile_params)

    # We always compute the training loss and metrics, at least once.
    training_metrics_final, training_metrics = get_metrics_sql_string(
        training_metrics, is_metrics_specified)
    training_loss_final, training_loss = get_metrics_sql_string(
        training_loss, True)

    # Validation loss and metrics are computed only if validation_table
    # is provided.
    if validation_set_provided:
        validation_metrics_final, validation_metrics = get_metrics_sql_string(
            validation_metrics, is_metrics_specified)
        validation_loss_final, validation_loss = get_metrics_sql_string(validation_loss)
        # Must quote the string before inserting to table. Explicitly
        # quoting it here since this can also take a NULL value, done
        # in the else part.
        validation_table = quote_literal(validation_table)
    else:
        validation_metrics = validation_loss = 'NULL'
        validation_metrics_final = validation_loss_final = 'NULL'
        validation_table = 'NULL'

    object_table = quote_literal(object_table) if object_table is not None else 'NULL'
    class_values_colnames = ' , '.join(class_values_colnames)
    if warm_start:
        plpy.execute("DROP TABLE {0}, {1}".format
                     (model, fit_validator.output_summary_model_table))
    create_output_summary_table = plpy.prepare("""
        CREATE TABLE {output_summary_model_table} AS
        SELECT
            $MAD${source_table}$MAD$::TEXT AS source_table,
            $MAD${model}$MAD$::TEXT AS model,
            ARRAY[{dep_name_list}]::TEXT[] AS dependent_varname,
            ARRAY[{ind_name_list}]::TEXT[] AS independent_varname,
            $MAD${model_arch_table}$MAD$::TEXT AS model_arch_table,
            {model_id}::INTEGER AS {model_id_colname},
            $1 AS compile_params,
            $2 AS fit_params,
            {num_iterations}::INTEGER AS num_iterations,
            {validation_table}::TEXT AS validation_table,
            {object_table}::TEXT AS object_table,
            {metrics_compute_frequency}::INTEGER AS metrics_compute_frequency,
            $3 AS name,
            $4 AS description,
            '{model_type}'::TEXT AS model_type,
            {model_size}::DOUBLE PRECISION AS model_size,
            '{start_training_time}'::TIMESTAMP AS start_training_time,
            '{end_training_time}'::TIMESTAMP AS end_training_time,
            $5 AS metrics_elapsed_time,
            '{version}'::TEXT AS madlib_version,
            ARRAY{num_classes}::INTEGER[] AS num_classes,
            ARRAY{dep_vartype}::TEXT[] AS {dependent_vartype_colname},
            {norm_const}::{FLOAT32_SQL_TYPE} AS {normalizing_const_colname},
            {metrics_type}::TEXT[] AS metrics_type,
            '{loss_type}'::TEXT AS loss_type,
            {training_metrics_final}::DOUBLE PRECISION AS training_metrics_final,
            {training_loss_final}::DOUBLE PRECISION AS training_loss_final,
            {training_metrics}::DOUBLE PRECISION[] AS training_metrics,
            {training_loss}::DOUBLE PRECISION[] AS training_loss,
            {validation_metrics_final}::DOUBLE PRECISION AS validation_metrics_final,
            {validation_loss_final}::DOUBLE PRECISION AS validation_loss_final,
            {validation_metrics}::DOUBLE PRECISION[] AS validation_metrics,
            {validation_loss}::DOUBLE PRECISION[] AS validation_loss,
            ARRAY{metrics_iters}::INTEGER[] AS metrics_iters,
            {class_values_colnames}
        FROM {source_summary_table}
        """.format(output_summary_model_table=fit_validator.output_summary_model_table,
                   dependent_vartype_colname=DEPENDENT_VARTYPE_COLNAME,
                   normalizing_const_colname=NORMALIZING_CONST_COLNAME,
                   FLOAT32_SQL_TYPE = FLOAT32_SQL_TYPE,
                   model_id_colname = ModelArchSchema.MODEL_ID,
                   source_summary_table=fit_validator.source_summary_table,
                   **locals()),
                                               ["TEXT", "TEXT", "TEXT", "TEXT", "DOUBLE PRECISION[]"])
    plpy.execute(create_output_summary_table,
                 [compile_params, fit_params, name,
                  description, metrics_elapsed_time])

    plpy.execute("""
        CREATE TABLE {0}
        (model_weights bytea,
        {1} json)""".format(model, ModelArchSchema.MODEL_ARCH))
    insert_output_table = plpy.prepare("""
        INSERT INTO {0} SELECT model_weights, {1}
        FROM (VALUES($1, $2))t(model_weights, {1})
        """.format(model, ModelArchSchema.MODEL_ARCH), ["bytea", "json"])
    plpy.execute(insert_output_table, [serialized_weights, model_arch])

    #TODO add a unit test for this in a future PR
    reset_cuda_env(original_cuda_env)


def fit_transition_gradients_batch_level(state, samples_every, dependent_var, independent_var,
                    dependent_var_shape,independent_var_shape,
                   compile_params, fit_params, dist_key, dist_key_mapping,
                   current_seg_id, segments_per_host, images_per_seg,
                   accessible_gpus_for_seg, prev_serialized_weights,
                   is_multiple_model=False, custom_function_map=None, **kwargs):
    """
    This transition function is common for madlib_keras_fit() and
    madlib_keras_fit_multiple_model(). The important difference between
    these two calls is the way tensorflow/keras sessions and GD gets used.
    For madlib_keras_fit_multiple_model,
        a. We create a tensorflow session per hop and store it in GD alongwith
        the model and clear both GD and the session at the end of each
        hop.
    For madlib_keras_fit,
        b. We create only one tensorflow session for both fit and eval transition
        functions and store it in GD. This session gets reused by both fit and eval
        and only gets cleared in eval transition at the last row of the last iteration.
    """
    record("seg {} enter fit_trans at {}\n".format(current_seg_id ,time.time()))
    if not dependent_var_shape[0] or not independent_var_shape[0] \
            or dependent_var[0] is None or independent_var[0] is None:
        plpy.error("fit_transition called with no data")

    if not prev_serialized_weights:
        return state

    master_id = 6
    current_seg_id = dist_key_mapping.index(dist_key)
    GD = kwargs['GD']

    trans_enter_time = time.time()
    segment_model = tf_get_init_model_and_sess(GD, current_seg_id, master_id)
    device_name = get_device_name_and_set_cuda_env(accessible_gpus_for_seg[current_seg_id], current_seg_id)
    model_id = segment_model._fetch_results("select max(model_id) from embed_model;")
    if GD_STORE.AGG_IMAGE_COUNT in GD:
        record("seg {} restore model at {}\n".format(current_seg_id, time.time()))
        agg_image_count = GD[GD_STORE.AGG_IMAGE_COUNT]
    else:
        record("seg {} receive model at {}\n".format(current_seg_id, time.time()))
        agg_image_count = 0
        GD[GD_STORE.AGG_IMAGE_COUNT] = agg_image_count
        segment_model.pull_dense_weights_udaf()
    record("seg {} finish set model at {}\n".format(current_seg_id, time.time()))

    '''if current_seg_id == master_id:
        record("seg {} start apply_embedding_grads at {}\n".format(current_seg_id, time.time()))
        segment_model.apply_embedding_grads()
        record("seg {} end apply_embedding_grads at {}\n".format(current_seg_id, time.time()))
        return state'''

    t1 = time.time()
    x_train = []
    y_train = []
    # Prepare the data

    for counter, shape in enumerate(independent_var_shape):
        x_train.append(np_array_float32(independent_var[counter], shape))

    for counter, shape in enumerate(dependent_var_shape):
        y_train.append(np_array_int16(dependent_var[counter], shape))

    # Fit segment model on data
    #TODO consider not doing this every time
    xi_tmp = x_train[0].T
    xv_tmp = x_train[1].T
    xi = [np.array(xi_tmp[i, :]) for i in range(xi_tmp.shape[0])]
    xv = [np.array(xv_tmp[i, :]) for i in range(xv_tmp.shape[0])]
    x_train_tmp = x_train[0].T

    record("Xi len: {}".format(len(xi[0])))
    record("Xv : {}\n".format(len(xv[0])))
    emd_id_unique = np.unique(np.array(xi))
    record("emd_id_unique : {}\n".format(emd_id_unique))
    embedding_id_mapping = segment_model.pull_embedding_weights(emd_id_unique)
    '''record("embedding_weights : {}\n".format(len(embedding_weights)))
    embedding_weights_transfer = list()
    embedding_bias_transfer = list()
    embedding_id_mapping = dict()
    for i in range(len(emd_id_unique)):
        embedding_weights_transfer.append(tf_deserialize_embedding(embedding_weights[i]))
        embedding_bias_transfer.append(tf_deserialize_embedding(embedding_bias[i]))
        embedding_id_mapping[emd_id_unique[i]] = i
    variables_ = list()
    variables_.append(np.array(embedding_weights_transfer))
    variables_.append(np.array(embedding_bias_transfer))

    feed_dict = dict()
    for i, placeholder in enumerate(segment_model.embed_update_placehoders):
        feed_dict[placeholder] = variables_[i]
    segment_model.sess.run(segment_model.embed_update_ops, feed_dict=feed_dict)
'''

    Xi_batch_local = np.vectorize(embedding_id_mapping.get)(xi)
    record("seg {} prepare last {}\n".format(current_seg_id, round(time.time() - t1,2)))

    record("seg {} train model at {}\n".format(current_seg_id, time.time()))
    return_grads = 1
    grads = segment_model.gradients_compute(Xi_batch_local.T, np.array(xv).T, y_train[0][:,0].reshape(-1, 1), return_grads)
    record("seg {} finish compute grads at {}\n".format(current_seg_id, time.time()))
    reverse_map = dict(zip(embedding_id_mapping.values(), embedding_id_mapping.keys()))
    model_worker.push_embedding_grads(grads[0:2],reverse_map)
    record("seg {} finish push embedding grads at {}\n".format(current_seg_id, time.time()))
    res = segment_model.evaluate_per_batch(Xi_batch_local.T, np.array(xv).T, y_train[0][:,0].reshape(-1, 1))
    record("seg {} metric : {}\n".format(current_seg_id, res))
    # Aggregating number of images, loss and accuracy
    record("seg {} finish update dense at {}\n".format(current_seg_id, time.time()))
    segment_model.version = segment_model.version + 1
    segment_model.model_id = segment_model.model_id + 1
    agg_image_count += dependent_var_shape[0][0]
    GD[GD_STORE.AGG_IMAGE_COUNT] = agg_image_count
    total_images = get_image_count_per_seg_from_array(dist_key_mapping.index(dist_key),
                                                      images_per_seg)
    is_last_row = 1==1
    record("seg {}: agg_image_count:{}, samples_every:{}\n".format(current_seg_id, agg_image_count, samples_every))
    return_state = get_grads_to_return(is_last_row, is_multiple_model,
                                       agg_image_count, agg_image_count)
    is_last_row = agg_image_count+samples_every > total_images
    if is_last_row:
        variables = segment_model.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[2:]
        variables_value = [segment_model.sess.run(v) for v in variables]
        flattened_weights = [np.float32(w).tostring() for w in variables_value]
        flattened_weights = "".join(flattened_weights)
        return_state = flattened_weights
        record("seg {}: enter last row\n".format(current_seg_id))
        del GD[GD_STORE.AGG_IMAGE_COUNT]  # Must be reset after each pass through images
        if is_multiple_model:
            GD_STORE.clear(GD)

    trans_exit_time = time.time()
    DEBUG.plpy.info("|_fit_transition_time_|{}|".format(trans_exit_time - trans_enter_time))
    record("seg {} leave fit_trans at {}\n".format(current_seg_id ,time.time()))
    record("seg {} fit_trans last {}\n".format(current_seg_id ,trans_exit_time - trans_enter_time))
    return return_state

def fit_transition_gradients_wide_batch_level(state, samples_every, dependent_var1, dependent_var2, dependent_var3,
                   dependent_var4, dependent_var5, independent_var1,
                   independent_var2, independent_var3, independent_var4,
                   independent_var5,
                   dependent_var_shape,independent_var_shape,
                   compile_params, fit_params, dist_key, dist_key_mapping,
                   current_seg_id, segments_per_host, images_per_seg,
                   accessible_gpus_for_seg, prev_serialized_weights,
                   is_multiple_model=False, custom_function_map=None, **kwargs):

    if not independent_var1 or not dependent_var1:
        return state
    dependent_var = [dependent_var1, dependent_var2, dependent_var3,
                        dependent_var4, dependent_var5]
    independent_var = [independent_var1, independent_var2, independent_var3,
                        independent_var4, independent_var5]

    dependent_var = [i for i in dependent_var if i is not None]
    independent_var = [i for i in independent_var if i is not None]

    return fit_transition_gradients_batch_level(state, samples_every, dependent_var,
                   independent_var,dependent_var_shape,independent_var_shape,
                   compile_params, fit_params, dist_key, dist_key_mapping,
                   current_seg_id, segments_per_host, images_per_seg,
                   accessible_gpus_for_seg, prev_serialized_weights,
                   is_multiple_model, custom_function_map, **kwargs)

def get_loss_metric_from_keras_eval_ctq(schema_madlib, table, dependent_varname,
                                    independent_varname, compile_params, fit_params,
                                    serialized_weights, use_gpus,
                                    accessible_gpus_for_seg, segments_per_host,
                                    dist_key_mapping, images_per_seg,
                                    should_clear_session=True, custom_function_map=None,
                                    model_table=None, mst_key=None):
    """
    This function will call the internal keras evaluate function to get the loss
    and accuracy of each tuple which then gets averaged to get the final result.
    """

    dist_key_col = '0' if is_platform_pg() else '__table__.{0}'.format(DISTRIBUTION_KEY_COLNAME)
    gp_segment_id_col = '0' if is_platform_pg() else '__table__.{0}'.format(GP_SEGMENT_ID_COLNAME)

    """
    This function will call the internal keras evaluate function to get the loss
    and accuracy of each tuple which then gets averaged to get the final result.
    """
    use_gpus = use_gpus if use_gpus else False

    mb_dep_var_cols_sql = ', '.join(dependent_varname)
    mb_indep_var_cols_sql = ', '.join(independent_varname)
    dep_shape_cols = [add_postfix(i, "_shape") for i in dependent_varname]
    ind_shape_cols = [add_postfix(i, "_shape") for i in independent_varname]
    dep_shape_cols_sql = ', '.join(dep_shape_cols)
    ind_shape_cols_sql = ', '.join(ind_shape_cols)

    eval_sql = """
        select ({schema_madlib}.internal_keras_evaluate(
                                            ARRAY[{mb_dep_var_cols_sql}],
                                            ARRAY[{mb_indep_var_cols_sql}],
                                            ARRAY[{dep_shape_cols_sql}],
                                            ARRAY[{ind_shape_cols_sql}],
                                            {weights},
                                            {compile_params},
                                            {fit_params},
                                            {dist_key_col},
                                            ARRAY{dist_key_mapping},
                                            {gp_segment_id_col},
                                            ARRAY{segments_per_host},
                                            ARRAY{images_per_seg},
                                            ARRAY{accessible_gpus_for_seg},
                                            {should_clear_session},
                                            {custom_map_var}
                                            )) as loss_metric
        from {table} AS __table__ {mult_sql}
        """

    if mst_key:
        weights = '__mt__.{0}'.format(MODEL_WEIGHTS_COLNAME)
        mst_key_col = ModelSelectionSchema.MST_KEY
        mult_sql = ', {model_table} AS __mt__ WHERE {mst_key_col} = {mst_key}'.format(**locals())
        custom_map_var = '$1'
        evaluate_query = plpy.prepare(eval_sql.format(**locals()), ["bytea"])
        res = plpy.execute(evaluate_query, [custom_function_map])
    else:
        weights = '$1'
        mult_sql = ''
        custom_map_var = '$2'
        evaluate_query = plpy.prepare(eval_sql.format(**locals()), ["bytea", "bytea"])
        res = plpy.execute(evaluate_query, [serialized_weights, custom_function_map])


    if res is None:
        plpy.error("Zero rows returned from evaluate query: {}".format(evaluate_query))
    else:
        loss_metric = res[0]['loss_metric']
    return loss_metric

def internal_keras_eval_transition_ctq(state, dependent_var, independent_var,
                                   dependent_var_shape, independent_var_shape,
                                   serialized_weights, compile_params,fit_params,
                                   dist_key, dist_key_mapping, current_seg_id,
                                   segments_per_host, images_per_seg,
                                   accessible_gpus_for_seg, should_clear_session,
                                   custom_function_map=None, **kwargs):
    GD = kwargs['GD']
    device_name = get_device_name_and_set_cuda_env(accessible_gpus_for_seg[current_seg_id], current_seg_id)

    """
    This transition function is common to evaluate as well as the fit functions.
    All these calls have a different logic for creating and clear the tensorflow
    session
    For evaluate,
        We create only one tensorflow session and store it in GD.
        should_clear_session is always set to true, so the session and GD is
        cleared once the last buffer is evaluated on each segment.
    For fit,
        We reuse the session and GD created as part of fit_transition and only clear
        the session and GD at last row of the last iteration of eval_transition.
        should_clear_session is only set to true for the last call to eval_transition
        which can be either the training eval or validation eval
    For fit_multiple,
        We create one session per hop and store it in GD.
        should_clear_session is always set to true, so the session and GD is
        cleared once the last buffer is evaluated on each segment.
    """

    multi_output = True if len(dependent_var) > 1 else False

    record("seg {} enter eval_trans at {}\n".format(current_seg_id ,time.time()))

    if multi_output:
        output_count = len(dependent_var)
        agg_loss = state[0]
        if agg_loss == 0:
            state = []
            for i in range(2*output_count+2):
                state.append(0)
        agg_image_count = state[-1]
        aux_losses = []
        aux_metrics = []
        for counter in range(output_count):
            aux_losses.append(state[2*counter+1])
            aux_metrics.append(state[2*counter+2])

    else:
        agg_loss, agg_metric, agg_image_count = state
    master_id = 0
    segment_model = tf_get_init_model_and_sess(GD, current_seg_id, master_id)

    record("seg {} restore eval model at {}\n".format(current_seg_id, time.time()))

    if not agg_image_count:
        # These should already be 0, but just in case make sure
        agg_metric = 0
        agg_loss = 0
        segment_model.use_serialize_dense_weights(serialized_weights)
    t1 = time.time()
    x_val = []
    y_val = []
    for counter, shape in enumerate(independent_var_shape):
        x_val.append(np_array_float32(independent_var[counter], shape))
    for counter, shape in enumerate(dependent_var_shape):
        y_val.append(np_array_int16(dependent_var[counter], shape))

    image_count = len(y_val[0])
    agg_image_count += image_count
    x_val_tmp = x_val[0].T
    X = [np.array(x_val_tmp[i, :]) for i in range(x_val_tmp.shape[0])]
    xi = list()
    xv = list()
    for i in range(39):
        xi.append(X[i])
    for i in range(39,78):
        xv.append(X[i])

    emd_id_unique = np.unique(np.array(xi))

    emb_id_mapping = segment_model.pull_embedding_weights(emd_id_unique)

    Xi_batch_local = np.vectorize(emb_id_mapping.get)(xi)
    record("seg {} prepare last {}\n".format(current_seg_id, round(time.time() - t1,2)))

    record("seg {} model eval at {}\n".format(current_seg_id, time.time()))
    res = segment_model.evaluate_per_batch(Xi_batch_local.T, np.array(xv).T, y_val[0][:,0].reshape(-1, 1))
    loss, metric = res
    record("loss_metric : {}, image_count = {}\n".format(res, image_count))
    metric = metric/image_count
    res = [loss, metric]
    record("seg {} model finish eval at {}\n".format(current_seg_id, time.time()))
    # if metric is None, model.evaluate will only return loss as a scalar
    # Otherwise, it will return a list which has loss and metric
    if multi_output:
        loss = res[0]
        agg_loss += (image_count * loss)
        for counter in range(output_count):
            # For multi output cases, res has the following structure
            # print(model.metrics_names)
            # ['loss', 'dense_4_loss', 'dense_5_loss', 'dense_4_acc', 'dense_5_acc']
            aux_losses[counter] = aux_losses[counter] + (image_count * res[counter+1])
            aux_metrics[counter] = aux_metrics[counter] + (image_count * res[counter+1+len(dependent_var)])
    else:
        if type(res) is list:
           loss, metric = res
        else:
            loss = 0
            metric = res

        agg_loss += (image_count * loss)
        agg_metric += (image_count * metric)

    total_images = get_image_count_per_seg_from_array(dist_key_mapping.index(dist_key),
                                                      images_per_seg)
    is_last_row = agg_image_count == total_images
    if is_last_row and should_clear_session:
        GD_STORE.clear(GD)
        del segment_model

    state = [agg_loss]

    if multi_output:
        for counter in range(output_count):
            state.append(aux_losses[counter])
            state.append(aux_metrics[counter])
    else:
        state.append(agg_metric)
    state.append(agg_image_count)
    record("seg {} finish eval_trans at {}\n".format(current_seg_id ,time.time()))
    return state

def internal_keras_eval_merge_ctq(state1, state2, **kwargs):
    record("mdw enter eval_merge at {}\n".format(time.time()))
    # If either state is None, return the other one
    if not state1 or not state2:
        return state1 or state2

    merged_state = []
    for i in range(len(state1)):
        merged_state.append(state1[i]+state2[i])
    record("mdw finish eval_merge at {}\n".format(time.time()))
    return merged_state

def internal_keras_eval_final_ctq(state, **kwargs):
    image_count = state[-1]

    if image_count == 0:
        plpy.error("internal_keras_eval_final: Total images processed is 0")

    for i in range(len(state)-1):
        state[i] = state[i]/image_count

    return state

from Vgg_net import VggNetModel_Worker,VggNetModel_Master,VggNetModel_Worker_new
import multiprocessing

def create_master(gpseg_list ,**kwargs):
    count = 0
    model_master = DeepFM_Master(**dfm_params)
    record("thread start create_master\n")
    dense_version = model_master.model_id
    work_embed_check = dict()
    dense_model_table = Schema.Dense_Model_Table
    for i in gpseg_list:
        work_embed_check[i] = 0
    while True:
        dense_query = '''SELECT model_id FROM {dense_model_table} WHERE worker_id = 6'''.format(**locals())
        dense_res = model_master._fetch_results(dense_query)
        if dense_res:
            for row in dense_res:
                model_version = row
                if model_version != dense_version:
                    model_master.apply_dense_weights()
                    dense_version = model_version
                    break
        t1 = time.time()
        embed_gradient_table = Schema.Embed_GRADIENT_TABLE
        embed_query = '''SELECT worker_id, model_id FROM {embed_gradient_table} order by model_id'''.format(**locals())
        embed_results = model_master._fetch_results(embed_query)
        model_version = 0
        if embed_results:
            for row in embed_results:
                worker_id, model_version = row
                if model_version >= work_embed_check[worker_id]:
                    work_embed_check[worker_id] = model_version
                    model_master.apply_embed_grads_per_worker(worker_id, model_version)
            t2 = time.time()
            model_master.save_embedding_with_cache()
            record("save embedding in DB last {}\n".format(time.time() - t2))
            model_master.embedding_dict.clear()
            #model_master.embedding_cache_dict.clear()
            #model_master.embedding_bias_cache_dict.clear()


def batch_compute_gradient(current_seg_id, batch_num, average_iter, source_table_name,**kwargs):
    master_id = 0
    avg_time = average_iter
    record("epoch_start with average_iter :{}\n".format(avg_time))
    model_worker = DeepFM_Worker(current_seg_id,**dfm_params)
    t1 = time.time()
    source_table = source_table_name
    flag = 0
    for batch_index in range(batch_num):
        if flag == 1:
            t0 = time.time()
            model_worker.pull_dense_weights()
            record("seg {} pull dense start last {}\n".format(current_seg_id,time.time()-t0))
            flag = 0
            if batch_index != 0:
                model_worker.version = model_worker.version + 1
                model_worker.model_id = model_worker.model_id + 1
        buffer_index = batch_index * 4 + current_seg_id
        select_query = '''  SELECT xi,xv,y,xi_shape,xv_shape,y_shape
                            FROM {source_table}
                            where {source_table}.buffer_id = {buffer_index} and gp_segment_id = {current_seg_id}'''.format(**locals())
        data = model_worker._fetch_results(select_query)
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

        emd_id_unique = np.unique(np.array(xi,dtype=np.int32))
        embed_id_mapping = model_worker.pull_embedding_weights(emd_id_unique)
        Xi_batch_local = np.vectorize(embed_id_mapping.get)(xi)
        record("seg {} prepare last {}\n".format(current_seg_id, round(time.time() - t1,2)))
        t2 = time.time()
        record("seg {} train model at {}\n".format(current_seg_id, t2))
        Xi_train = Xi_batch_local.T
        Xv_train = np.array(xv).T
        Y_train = list()
        for y in y_train:
            Y_train.append(y[0])
        Y_train = np.array(Y_train).reshape(-1,1)
        return_grads = 1
        if batch_index % 10 == 0:
            return_grads = 1
        else:
            return_grads = 0
        grads = model_worker.gradients_compute(Xi_train, Xv_train, Y_train, return_grads)
        record("seg {} train model last {}\n".format(current_seg_id, round(time.time() - t1,2)))
        if return_grads:
            res = model_worker.evaluate_per_batch(Xi_train, Xv_train, Y_train)
            record("seg {} evluate at batch {} with res {}\n".format(current_seg_id, batch_index, res))
            reverse_map = dict(zip(embed_id_mapping.values(), embed_id_mapping.keys()))
            model_worker.push_embedding_grads(grads[0:2],reverse_map)
            record("seg {} push_embedding_grads at {}\n".format(current_seg_id, time.time()))
            flattened_dense_grads = model_worker.update_dense(grads[2:])
        if batch_index % avg_time == 0 or batch_index == batch_num-1:
            model_worker.push_dense_weights()
            flag = 1
        else:
            model_worker.version = model_worker.version + 1
            model_worker.model_id = model_worker.model_id + 1

def batch_compute_gradient_hot_key(current_seg_id, batch_num, average_iter, source_table_name, **kwargs):
    master_id = 0
    avg_time = average_iter
    record("epoch_start with average_iter :{}\n".format(avg_time))
    model_worker = DeepFM_Worker(current_seg_id,**dfm_params)
    model_worker.hot_key_init()
    t1 = time.time()
    hash_time = 10
    source_table = source_table_name
    flag = 0
    batch_list = list()
    sql = '''select buffer_id from {source_table} where gp_segment_id = {current_seg_id}'''.format(**locals())
    list_b = model_worker._fetch_results(sql)
    for row in list_b:
        batch_list.append(row[0])
    buffer_index = 0
    for batch_index in batch_list:
        if flag == 1:
            t0 = time.time()
            model_worker.pull_dense_weights()
            record("seg {} pull dense start last {}\n".format(current_seg_id,time.time()-t0))
            flag = 0
            if batch_index != 0:
                model_worker.version = model_worker.version + 1
                model_worker.model_id = model_worker.model_id + 1
        buffer_index = buffer_index+1
        select_query = '''  SELECT xi,xv,y,xi_shape,xv_shape,y_shape
                            FROM {source_table}
                            where {source_table}.buffer_id = {batch_index} and gp_segment_id = {current_seg_id}'''.format(**locals())
        data = model_worker._fetch_results(select_query)
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

        emd_id_unique = np.unique(np.array(xi,dtype=np.int32))
        model_worker.key_time_check(hash_time)
        embed_id_mapping = model_worker.pull_new_key(emd_id_unique)
        #embed_id_mapping = model_worker.pull_embedding_weights(emd_id_unique)
        Xi_batch_local = np.vectorize(embed_id_mapping.get)(xi)
        record("seg {} prepare last {}\n".format(current_seg_id, round(time.time() - t1,2)))
        t2 = time.time()
        Xi_train = Xi_batch_local.T
        Xv_train = np.array(xv).T
        Y_train = list()
        for y in y_train:
            Y_train.append(y[0])
        Y_train = np.array(Y_train).reshape(-1,1)
        return_grads = 1
        if buffer_index % 10 == 0:
            return_grads = 1
        else:
            return_grads = 0
        grads = model_worker.gradients_compute(Xi_train, Xv_train, Y_train, return_grads)
        record("seg {} train model last {}\n".format(current_seg_id, round(time.time() - t1,2)))
        if return_grads:
            res = model_worker.evaluate_per_batch(Xi_train, Xv_train, Y_train)
            record("seg {} evluate at batch {} with res {}\n".format(current_seg_id, batch_index, res))
            reverse_map = dict(zip(embed_id_mapping.values(), embed_id_mapping.keys()))
            model_worker.push_embedding_grads(grads[0:2],reverse_map)
            record("seg {} push_embedding_grads at {}\n".format(current_seg_id, time.time()))
            flattened_dense_grads = model_worker.update_dense(grads[2:])
        if buffer_index % avg_time == 0 or buffer_index == batch_num-1:
            model_worker.push_dense_weights()
            flag = 1
        else:
            model_worker.version = model_worker.version + 1
            model_worker.model_id = model_worker.model_id + 1


def buffer_preload(source_table_name, block_all, **kwargs):
    seg_ip = '127.0.0.1'
    port = 6000
    user = 'gpadmin'
    db = 'gpadmin'
    conn = p2.connect(host=seg_ip, user=user, dbname=db, port=port, options='-c gp_session_role=utility')
    block_check_sql = '''SELECT count(*) AS buffers
FROM pg_buffercache b
INNER JOIN pg_class c ON b.relfilenode = c.relfilenode
WHERE c.relname = 'embed_model'
GROUP BY c.relname;'''
    cursor = conn.cursor()
    cursor.execute(block_check_sql)
    block_num = cursor.fetchall()
    if block_num:
        block_num = block_num[0][0]
    else :
        buffer_clear_sql = '''select pg_prewarm('{}')'''.format(source_table_name)
        cursor = conn.cursor()
        cursor.execute(buffer_clear_sql)
        conn.commit()
        return 
    if float(block_num) > 100000 :
        conn.commit()
    else:
        buffer_clear_sql = '''select pg_prewarm('{}')'''.format(source_table_name)
        cursor = conn.cursor()
        cursor.execute(buffer_clear_sql)
        conn.commit()

def get_seg_id(e_id,**kwargs):
    dist_id = e_id % 6
    if dist_id == 0:
        target_seg = 1
    elif dist_id == 1:
        target_seg = 4
    elif dist_id == 2:
        target_seg = 3
    elif dist_id == 3:
        target_seg = 0
    elif dist_id == 4:
        target_seg = 5
    else:
        target_seg = 2
    return target_seg

def get_block_into_dbcache(id_list,**kwargs):
    t0 = time.time()
    ip_list = ['172.17.31.86','172.17.31.91','172.17.31.85','172.17.31.92','172.17.31.90','172.17.31.88']
    sql = "select id,block_number from block_hash where id in {}".format(tuple(id_list))
    seg_ip = '172.17.31.87'
    port = 5432
    user = 'gpadmin'
    db = 'gpadmin'
    conn = p2.connect(host=seg_ip, user=user, dbname=db, port=port)
    cursor = conn.cursor()
    cursor.execute(sql)
    res = cursor.fetchall()
    record("prewarm with block at {}\n".format(time.time() - t0))
    conn.commit()
    cnt = 0
    blk0 = dict()
    blk1 = dict()
    blk2 = dict()
    blk3 = dict()
    blk4 = dict()
    blk5 = dict()
    for e_id,data in res:
        target_seg = get_seg_id(e_id)
        if target_seg == 0:
            blk0[data] = 1
        elif target_seg == 1:
            blk1[data] = 1
        elif target_seg == 2:
            blk2[data] = 1
        elif target_seg == 3:
            blk3[data] = 1
        elif target_seg == 4:
            blk4[data] = 1
        elif target_seg == 5:
            blk5[data] = 1
    record("block done , start prewarm at {}\n".format(time.time() - t0))
    sql0 = "select pg_prewarm_blocks('embed_model','buffer','main',ARRAY{})".format(list(blk0.keys()))
    conn0 = p2.connect(host=ip_list[0], user=user, dbname=db, port=6000, options='-c gp_session_role=utility')
    cursor0 = conn0.cursor()
    cursor0.execute(sql0)
    res = cursor0.fetchall()
    record("seg 0 prewarm {} in {} at {}\n".format(res,len(list(blk0.keys())),time.time() - t0))
    conn0.commit()
    sql1 = "select pg_prewarm_blocks('embed_model','buffer','main',ARRAY{})".format(list(blk1.keys()))
    conn1 = p2.connect(host=ip_list[1], user=user, dbname=db, port=6000, options='-c gp_session_role=utility')
    cursor1 = conn1.cursor()
    cursor1.execute(sql1)
    res = cursor1.fetchall()
    record("seg 1 prewarm {} in {} at {}\n".format(res,len(list(blk1.keys())),time.time() - t0))
    conn1.commit()
    sql2 = "select pg_prewarm_blocks('embed_model','buffer','main',ARRAY{})".format(list(blk2.keys()))
    conn2 = p2.connect(host=ip_list[2], user=user, dbname=db, port=6000, options='-c gp_session_role=utility')
    cursor2 = conn2.cursor()
    cursor2.execute(sql2)
    res = cursor2.fetchall()
    record("seg 2 prewarm {} in {} at {}\n".format(res,len(list(blk2.keys())),time.time() - t0))
    conn2.commit()
    sql3 = "select pg_prewarm_blocks('embed_model','buffer','main',ARRAY{})".format(list(blk3.keys()))
    conn3 = p2.connect(host=ip_list[3], user=user, dbname=db, port=6000, options='-c gp_session_role=utility')
    cursor3 = conn3.cursor()
    cursor3.execute(sql3)
    res = cursor3.fetchall()
    record("seg 3 prewarm {} in {} at {}\n".format(res,len(list(blk3.keys())),time.time() - t0))
    conn3.commit()
    sql4 = "select pg_prewarm_blocks('embed_model','buffer','main',ARRAY{})".format(list(blk4.keys()))
    conn4 = p2.connect(host=ip_list[4], user=user, dbname=db, port=6000, options='-c gp_session_role=utility')
    cursor4 = conn4.cursor()
    cursor4.execute(sql4)
    res = cursor4.fetchall()
    record("seg 4 prewarm {} in {} at {}\n".format(res,len(list(blk4.keys())),time.time() - t0))
    conn4.commit()
    sql5 = "select pg_prewarm_blocks('embed_model','buffer','main',ARRAY{})".format(list(blk5.keys()))
    conn5 = p2.connect(host=ip_list[5], user=user, dbname=db, port=6000, options='-c gp_session_role=utility')
    cursor5 = conn5.cursor()
    cursor5.execute(sql5)
    res = cursor5.fetchall()
    record("seg 5 prewarm {} in {} at {}\n".format(res,len(list(blk5.keys())),time.time() - t0))
    conn5.commit()

def buffer_clear(source_table_name, **kwargs):
    seg_ip = '127.0.0.1'
    port = 6000
    user = 'gpadmin'
    db = 'gpadmin'
    conn = p2.connect(host=seg_ip, user=user, dbname=db, port=port, options='-c gp_session_role=utility')
    buffer_clear_sql = '''select pg_drop_rel_cache('{}')'''.format(source_table_name)
    cursor = conn.cursor()
    cursor.execute(buffer_clear_sql)
    conn.commit()


import psutil
def batch_compute_gradient_prefetch(current_seg_id, batch_num, average_iter, source_table_name, **kwargs):
    master_id = 0
    avg_time = average_iter
    record("epoch_start with average_iter :{}\n".format(avg_time))
    model_worker = DeepFM_Worker(current_seg_id, **dfm_params)
    t1 = time.time()
    source_table = source_table_name
    parameter_block = '''select relpages from pg_class where relname = 'embed_model';'''
    block_all = model_worker._fetch_results_onseg(parameter_block)[0][0]
    record("parameter_block num : {}\n".format(block_all))
    # vectorization data train:
    '''flag = 0
    prefetch_D = 10
    prefetch_size = 102400
    prefetch_time = model_worker.total_sample_worker/prefetch_size
    for index in range(prefetch_time):
        t0 = time.time()
        model_worker.pull_dense_weights()
        record("seg {} pull dense start last {}\n".format(current_seg_id,time.time()-t0))
        df = model_worker.prefetch_feature_id(prefetch_size,index)
        emd_id_unique = np.unique(np.concatenate(df['xi']))
        embed_id_mapping = model_worker.pull_embedding_weights(emd_id_unique)
        record("seg {} prepare last {}\n".format(current_seg_id, round(time.time() - t1,2)))
        for i in range(prefetch_D):
            xi, xv, y = model_worker.get_batch_data_block(prefetch_size/prefetch_D, index*prefetch_D+i)
            record("data fetch time at {}\n".format(time.time() - t1))
            Xi_batch_local = np.vectorize(embed_id_mapping.get)(np.array(xi))
            return_grads = 1
            grads = model_worker.gradients_compute(Xi_batch_local, xv, y, return_grads)
            grads = model_worker.gradient_transform(grads,embed_id_mapping)
            record("seg {} train model last {} with {}\n".format(current_seg_id, round(time.time() - t1,2),len(grads)))
            if return_grads:
                res = model_worker.evaluate_per_batch(Xi_batch_local, xv, y)
                record("seg {} evluate at batch {} with res {}\n".format(current_seg_id, buffer_index, res))
                model_worker.push_embedding_grads_dps(0,grads[0:2])
                record("seg {} push_embedding_grads at {}\n".format(current_seg_id, round(time.time()-t1,2)))
            else:
                model_worker.version = model_worker.version + 1
                model_worker.model_id = model_worker.model_id + 1
        model_worker.push_dense_weights()'''
    # buffer level train:
    prefetch_D = 5
    flag = 0
    batch_list = list()
    sql = "select buffer_id from {source_table} where gp_segment_id = {current_seg_id}".format(**locals())
    list_b = model_worker._fetch_results(sql)
    for row in list_b:
        batch_list.append(row[0])
    buffer_index = 0
    while buffer_index < batch_num:
        if flag == 1:
            t0 = time.time()
            model_worker.pull_dense_weights()
            record("seg {} pull dense start last {}\n".format(current_seg_id,time.time()-t0))
            flag = 0
            if buffer_index != 0:
                model_worker.version = model_worker.version + 1
                model_worker.model_id = model_worker.model_id + 1
        batch_temp = tuple(batch_list[buffer_index:buffer_index + prefetch_D])
        #x% DPS
        feature_id_query = '''SELECT xi,xi_shape
                            FROM {source_table}
                            where gp_segment_id = {current_seg_id} and {source_table}.buffer_id in {batch_temp}'''.format(**locals())
        data = model_worker._fetch_results_onseg(feature_id_query)
        xi = list()
        xi_shape = list()
        for i, row in enumerate(data):
            xi.append(row[0])
            xi_shape.append(row[1])
        xi_train = list()
        for i in range(prefetch_D):
            xi_train.append(np_array_float32(xi[i],xi_shape[i]))
        xi_train = np.concatenate((xi_train),axis = 0)
        xi_tmp = xi_train.T
        xi = [np.array(xi_tmp[i, :]) for i in range(xi_tmp.shape[0])]
        embed_id_unique = np.unique(np.array(xi,dtype=np.int32))
        '''end_id = int(0.35*(len(embed_id_unique)))
        record("end_id = {}\n".format(end_id))
        embedding_id_prefetch = embed_id_unique[0:end_id]
        model_worker.hot_id = embedding_id_prefetch
        model_worker.hot_key_init()'''
        #100%prefetch
        embed_id_mapping = model_worker.pull_embedding_weights(embed_id_unique)
        #batch_temp = batch_list[buffer_index]
        for buffer_id in batch_temp:
            select_query = '''SELECT xi,xv,y,xi_shape,xv_shape,y_shape
                                FROM {source_table}
                                where gp_segment_id = {current_seg_id} and {source_table}.buffer_id = {buffer_id}'''.format(**locals())
            data = model_worker._fetch_results_onseg(select_query)
            buffer_clear(source_table_name)
            record("data fetch time at {}\n".format(time.time() - t1))
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
            xi_train = list()
            xv_train = list()
            y_train = list()
            for i in range(1):
                xi_train.append(np_array_float32(xi[i],xi_shape[i]))
                xv_train.append(np_array_float32(xv[i],xv_shape[i]))
                y_train.append(np_array_int16(y[i],target_shape[i]))
            xi_train = np.concatenate((xi_train),axis = 0)
            xv_train = np.concatenate((xv_train),axis = 0)
            y_train = np.concatenate((y_train),axis = 0)
            xi_tmp = xi_train.T
            xv_tmp = xv_train.T
            xi = [np.array(xi_tmp[i, :]) for i in range(xi_tmp.shape[0])]
            xv = [np.array(xv_tmp[i, :]) for i in range(xv_tmp.shape[0])]
            record("data prepare time {}\n".format(time.time() - t1))
            emd_id_unique = np.unique(np.array(xi,dtype=np.int32))
            #model_worker.key_time_check(hash_time)
            #embed_id_mapping = model_worker.pull_new_key(emd_id_unique)
            #buffer_preload('embed_model',block_all)
            #get_block_into_dbcache(emd_id_unique)
            record("parameter prefetch at {}\n".format(time.time() - t1))
            #embed_id_mapping = model_worker.hot_key_update(emd_id_unique)
            Xi_batch_local = np.vectorize(embed_id_mapping.get)(xi)
            record("seg {} prepare last {}\n".format(current_seg_id, round(time.time() - t1,2)))
            t2 = time.time()
            Xi_train = Xi_batch_local.T
            Xv_train = np.array(xv).T
            Y_train = list()
            for y in y_train:
                Y_train.append(y[0])
            Y_train = np.array(Y_train).reshape(-1,1)
            del y_train
            del xi_train
            del xv_train
            return_grads = 1
            grads = model_worker.gradients_compute(Xi_train, Xv_train, Y_train, return_grads)
            grads = model_worker.gradient_transform(grads,embed_id_mapping)
            record("seg {} train model last {} with {}\n".format(current_seg_id, round(time.time() - t1,2),len(grads)))
            if return_grads:
                res = model_worker.evaluate_per_batch(Xi_train, Xv_train, Y_train)
                record("seg {} evluate at batch {} with res {}\n".format(current_seg_id, buffer_index, res))
                del Xi_train
                del Xv_train
                del Y_train
                model_worker.push_embedding_grads_dps(0,grads[0:2])
                record("seg {} push_embedding_grads at {}\n".format(current_seg_id, round(time.time()-t1,2)))
                #flattened_dense_grads = model_worker.update_dense(grads[2:])
        if buffer_index % avg_time == 0 or buffer_index >= batch_num-1:
            model_worker.push_dense_weights()
            flag = 1
        else:
            model_worker.version = model_worker.version + prefetch_D
            model_worker.model_id = model_worker.model_id + prefetch_D
 
def dense_model_average(gpseg_list, average_iter, **kwargs):
    record("thread start dense_model_average\n")
    work_dense_check = dict()
    for i in gpseg_list:
        work_dense_check[i] = 0
    dense_version = 1
    model_version = 0
    conn = p2.connect(host='127.0.0.1', user='gpadmin', dbname='gpadmin', port='5432')
    cursor = conn.cursor()
    while True:
        dense_model_table = Schema.Dense_Model_Table
        dense_query = '''SELECT worker_id, model_id FROM {dense_model_table} WHERE worker_id < 6'''.format(**locals())
        cursor.execute(dense_query)
        dense_results = cursor.fetchall()
        if dense_results:
            for row in dense_results:
                worker_id, model_version = row
                if model_version != dense_version and work_dense_check[worker_id] == 0:
                    record("[Master] dense version : {}, push worker : {}\n".format(model_version, worker_id))
                    work_dense_check[worker_id] = 1
            if 0 not in work_dense_check.values():
                dense_avg_save(model_version)
                dense_version = model_version
                for i in gpseg_list:
                    work_dense_check[i] = 0


def dense_avg_save(model_version, **kwargs):
    dense_model_table = Schema.Dense_Model_Table
    conn = p2.connect(host='127.0.0.1', user='gpadmin', dbname='gpadmin', port='5432')
    cursor = conn.cursor()
    worked_id_list = [0,1,2,3,4,5]
    weight_transfer = list()
    for i in worked_id_list:
        dense_fetch = '''SELECT weight FROM {dense_model_table} WHERE model_id={model_version} and worker_id = {i}'''.format(**locals())
        cursor.execute(dense_fetch)
        results = cursor.fetchall()
        weight_transfer.append(tf_deserialize_1d_weights(results[0][0]))
    weights_avg = np.array(weight_transfer).mean(axis = 0)
    serialized_weights = tf_serialize_1d_weights(weights_avg)
    update_sql = '''UPDATE {dense_model_table} SET (model_id, weight) = ({model_version}, %s) where worker_id = 6'''.format(**locals())
    cursor.execute(update_sql, (p2.Binary(serialized_weights),))
    record("[Master] Save Dense at [{}]\n".format(time.time()))
    conn.commit()


def dense_subnet_compute(current_seg_id, batch_num, source_table_name, **kwargs):
    model_worker = VggNetModel_Worker_new(current_seg_id,num_classes = 100, seg_num=1)
    #model_worker.dense_load()
    t1 = time.time()
    source_table = source_table_name
    flag = 0
    #model_worker.dense_load()
    for batch_index in range(batch_num):
        buffer_index = batch_index * 4 + current_seg_id
        select_query = '''  SELECT x, y, x_shape, y_shape
                               FROM {source_table}
                               where {source_table}.buffer_id = {buffer_index} and gp_segment_id = {current_seg_id}'''.format(
            **locals())
        data = model_worker._fetch_results(select_query)
        x = list()
        target = list()
        x_shape = list()
        target_shape = list()
        for i, row in enumerate(data):
            x.append(row[0])
            target.append(row[1])
            x_shape.append(row[2])
            target_shape.append(row[3])
        x_train = np_array_float32(x[0], x_shape[0])
        y_train = np_array_int16(target[0], target_shape[0])
        x_tmp = x_train.T
        x = [np.array(x_tmp[i, :]) for i in range(x_tmp.shape[0])]
        X_train = np.array(x).T
        '''for i in range(len(X_train)):
            record("shape:{}\n".format(X_train[i].shape))
            img = X_train[i]
            img = img.flatten().reshape((32, 32, 3))
            X_train[i] = img'''
        X_train = np.reshape(X_train, (-1, 224, 224, 3))
        Y_train = y_train
        '''for y in y_train:
            y_train_onehot = tf.keras.utils.to_categorical(y[0], 10)
            record("{}\n".format(y))
            Y_train.append(y_train_onehot)'''
        Y_train = np.array(Y_train).reshape(-1, 100)
        loss = model_worker.train(X_train, Y_train)
        metric = model_worker.evaluate(X_train, Y_train)
        record("Worker {} batch {} with loss :{},metric:{}\n".format(current_seg_id,batch_index,loss,metric))
    model_worker.push_dense_weights()
    record("Worker {} push the subnet at {}\n".format(model_worker.worker_id,time.time()))
    return


def dense_distribution(clean, source_table_name, **kwargs):
    model_master = VggNetModel_Master(num_classes=100)
    source_table = source_table_name
    if clean:
        model_master.fn_partition(seg_num=4)
    else:
        model_master.redistribution(seg_num=4)
        buffer_index = np.random.randint(low = 0,high = 50)
        select_query = ''' SELECT x, y, x_shape, y_shape
                               FROM {source_table}
                               where {source_table}.buffer_id = {buffer_index}'''.format(
            **locals())
        data = model_master._fetch_results(select_query)
        x = list()
        target = list()
        x_shape = list()
        target_shape = list()
        for i, row in enumerate(data):
            x.append(row[0])
            target.append(row[1])
            x_shape.append(row[2])
            target_shape.append(row[3])
        x_train = np_array_float32(x[0], x_shape[0])
        y_train = np_array_int16(target[0], target_shape[0])
        x_tmp = x_train.T
        x = [np.array(x_tmp[i, :]) for i in range(x_tmp.shape[0])]
        X_train = np.array(x).T
        X_train = np.reshape(X_train, (-1, 224, 224, 3))
        Y_train = y_train
        Y_train = np.array(Y_train).reshape(-1, 100)
        loss,metric = model_master.evaluate(X_train, Y_train)
        record("Master loss,metric is {},{}\n".format(loss,metric))
        model_master.fn_partition(seg_num=4)
    return


def memory_check(ip, seg_port, sql_mem, **kwargs):
    user = 'gpadmin'
    dbname = 'gpadmin'
    mem_check_sql = "SELECT madlib.get_available_memory();"
    conn = p2.connect(host=ip, user=user, dbname=dbname, port=seg_port, options='-c gp_session_role=utility')
    cursor = conn.cursor()
    availiable_mem = cursor.execute(mem_check_sql)
    if availiable_mem > sql_mem:
        return True
    else:
        return False


def thread_computing(seg_ip, seg_id, seg_port, batch_index, source_table, is_average,asy_or_sy):
    user = 'gpadmin'
    dbname = 'gpadmin'
    if asy_or_sy == 1:
        sql = "select madlib.batch_compute_gradient_prefetch({seg_id}, {batch_index}, {is_average}, '{source_table}');".format(**locals())
    else:
        sql = "select madlib.dense_subnet_compute({seg_id}, {batch_index}, '{source_table}');".format(**locals())
    conn = p2.connect(host=seg_ip, user=user, dbname=dbname, port=seg_port, options='-c gp_session_role=utility')
    cursor = conn.cursor()
    cursor.execute(sql)

def thread_master(seg_ip,seg_port):
    user = 'gpadmin'
    dbname = 'gpadmin'
    seg_list = [0,1,2,3,4,5]
    sql = "select madlib.create_master(Array{seg_list});".format(**locals())
    conn = p2.connect(host=seg_ip, user=user, dbname=dbname, port=seg_port)
    cursor = conn.cursor()
    cursor.execute(sql)

def thread_dense(seg_ip,seg_port,avg_iter):
    user = 'gpadmin'
    dbname = 'gpadmin'
    seg_list = [0,1,2,3,4,5]
    sql = "select madlib.dense_model_average(Array{seg_list},{avg_iter});".format(**locals())
    conn = p2.connect(host=seg_ip, user=user, dbname=dbname, port=seg_port)
    cursor = conn.cursor()
    cursor.execute(sql)
#select madlib.control_start(Array[0,1,2,3],Array[33000,33001,33002,33003],1,800,100,'criteo_tensor_packed',1,10);
#select madlib.control_start(Array[0,1,2,3],Array[33000,33001,33002,33003],1,250,5,'imagenet_train_packed',0,10);
def control_start(seg_list, seg_port, sql_mem_use, batch_index, is_average, source_table_name, asy_or_sy, epoch, **kwargs):
    user = 'gpadmin'
    dbname = 'gpadmin'
    ip = '127.0.0.1'
    if asy_or_sy == 1:
        t0 = time.time()
        record("compute start at {}!/n".format(t0))
        '''thread_master_ = multiprocessing.Process(target=thread_master, args=(
                    ip, 5432),name='master_{}'.format(epoch))
        thread_master_.start()
        time.sleep(60)
        thread_dense_ = multiprocessing.Process(target=thread_dense, args=(
                ip, 5432, is_average),name='dense_{}'.format(epoch))
        thread_dense_.start()'''
        for i in range(epoch):
            threading_dict = {}
            for seg_id in range(len(seg_list)):
                seg_ip_prefix = '172.17.31.'
                seg_ip = seg_ip_prefix + str(seg_list[seg_id])
                port = seg_port[seg_id]
                thread = threading.Thread(target=thread_computing, args=(
                    seg_ip, seg_id, port, batch_index, source_table_name, is_average, asy_or_sy),
                                        name='training_{}'.format(seg_id))
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
            record("*"*50 + "\n")
            record("epoch {} end;\n".format(i))
        record("compute end last {}!/n".format(time.time()-t0))
        thread_dense_.join()
        thread_master_.join()
    else:
        fit_clean = 1
        for i in range(epoch):
            dense_distribution(clean=fit_clean,source_table_name=source_table_name)
            fit_clean = 0
            threading_dict = {}
            for seg_id in range(len(seg_list)):
                port = seg_port[seg_id]
                thread = threading.Thread(target=thread_computing, args=(
                    ip, seg_id, port, batch_index, source_table_name, is_average, asy_or_sy),
                                        name='training_{}'.format(seg_id))
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
