import tensorflow as tf

from reinforceable.types import NestedTensor 
from reinforceable.timestep import Timestep 


NestedTensor = NestedTensor|Timestep


def expand_dim(inputs: NestedTensor, axis: int) -> NestedTensor:
    return tf.nest.map_structure(lambda x: tf.expand_dims(x, axis), inputs)

def squeeze_dim(inputs: NestedTensor, axis: int) -> NestedTensor:
    return tf.nest.map_structure(lambda x: tf.squeeze(x, axis), inputs)

def get_size(inputs: NestedTensor, dim: int) -> NestedTensor:
    inputs = tf.nest.flatten(inputs)[0]
    return inputs.shape[dim] or tf.shape(inputs)[0]

def transpose_batch_time(
    inputs: NestedTensor,
    expand_composites: bool = False
) -> NestedTensor:

    def transpose_fn(x):
        if x is None:
            return None
        return tf.transpose(
            x, tf.concat(([1, 0], tf.range(2, tf.rank(x))), axis=0))
    
    return tf.nest.map_structure(
        transpose_fn, inputs, expand_composites=expand_composites)

def flatten_batch_time(
    inputs: NestedTensor,
    expand_composites: bool = False
) -> NestedTensor:

    def flatten_fn(x):
        if x is None:
            return None
        return tf.reshape(x, tf.concat([[-1], tf.shape(x)[2:]], axis=0))
    
    return tf.nest.map_structure(
        flatten_fn, inputs, expand_composites=expand_composites)

def unflatten_batch_time(
    inputs: NestedTensor,
    time_major: bool = True,
    batch_size: int = None,
    time_size: int = None,
    expand_composites: bool = False
) -> NestedTensor:

    if batch_size is None and time_size is None:
        raise ValueError(
            'Either `batch_size` or `time_size` needs to be known.')

    def unflatten_fn(x):
        if x is None:
            return None
        if batch_size is not None:
            shape = [-1, batch_size] if time_major else [batch_size, -1]
        else:
            shape = [time_size, -1] if time_major else [-1, time_size]
        return tf.reshape(
            x, tf.concat([shape, tf.shape(x)[1:]], axis=0))
    
    return tf.nest.map_structure(
        unflatten_fn, inputs, expand_composites=expand_composites)
