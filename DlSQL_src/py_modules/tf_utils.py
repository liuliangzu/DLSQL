import pdb

import numpy as np

from tensorflow.python.framework.ops import IndexedSlicesValue
import tensorflow as tf

def record(content):
    f = open("/data2/ruike/pg/madlib_model.sql", 'a')
    f.write(content)
    f.close()

def tf_serialize_gradient(grads):
    typeL, dense_shapeL = list(), list()
    for i in range(len(grads)):
        if isinstance(grads[i], IndexedSlicesValue):
            typeL.append('IndexedSlicesValue')
            dense_shapeL.append(grads[i].dense_shape.tolist())
            grads[i] = np.hstack((grads[i].values, grads[i].indices.reshape(-1,1)))
        else:
            typeL.append('ndarray')
            dense_shapeL.append([])

    shapes = [grad.shape for grad in grads]
    auxiliaries = [typeL, dense_shapeL]
    return tf_serialize_nd_weights(grads), shapes, auxiliaries

'''def tf_serialize_gradient(grads,embed_id):
    typeL, dense_shapeL = list(), list()
    for i in range(len(grads)):
        if isinstance(grads[i], IndexedSlicesValue):
            typeL.append('IndexedSlicesValue')
            dense_shapeL.append(grads[i].dense_shape.tolist())
            indices_new = np.vectorize(embed_id.get)(np.array(grads[i].indices))
            grads[i] = np.hstack((grads[i].values, np.array(indices_new).reshape(-1,1)))
        else:
            typeL.append('ndarray')
            dense_shapeL.append([])

    shapes = [grad.shape for grad in grads]
    auxiliaries = [typeL, dense_shapeL]
    return tf_serialize_nd_weights(grads), shapes, auxiliaries'''

def tf_deserialize_gradient(grads_serialized, shapes, auxiliaries=None):
    if auxiliaries is None:
        return tf_deserialize_as_nd_weights(grads_serialized, shapes)
    else:
        typeL, dense_shapeL = auxiliaries

    gradients = tf_deserialize_as_nd_weights(grads_serialized, shapes)
    for i, type_ in enumerate(typeL):
        if type_ == 'IndexedSlicesValue':
            values = gradients[i][:,:-1]
            indices = gradients[i][:,-1]
            gradients[i] = IndexedSlicesValue(values, indices, dense_shapeL[i])

    return gradients


def tf_serialize_embedding(weights):
    if weights is None:
        return None
    weights = np.float32(weights).tostring()
    flattened_weights = "".join(weights)
    return flattened_weights


def tf_deserialize_embedding(weights):
    model_weights_serialized = np.fromstring(weights, dtype=np.float32)
    return model_weights_serialized



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


def tf_deserialize_1d_weights(model_weights):
    return_state = np.fromstring(model_weights, dtype=np.float32)
    return return_state


def tf_serialize_1d_weights(model_weights):
    weights = np.float32(model_weights).tostring()
    return_state = "".join(weights)
    return return_state


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
