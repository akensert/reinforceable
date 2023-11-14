import tensorflow as tf
from keras import layers

from reinforceable.utils import nested_ops


class StatefulRNN(layers.Layer):
    
    '''A stateful recurrent neural network.'''

    def __init__(
        self, 
        cell: layers.GRUCell|layers.LSTMCell, 
        time_major: bool = True,
        parallel_iterations: int = 20, 
        swap_memory: bool = None, 
        name: str = 'StatefulRNN',
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)

        self.cell = cell
        self._state = None

        self._time_major = time_major
        self._parallel_iterations = parallel_iterations
        self._swap_memory = swap_memory
        
    def call(
        self, 
        inputs: tf.Tensor, 
        states_mask: tf.Tensor = None, 
    ):
        if not self._time_major:
            inputs = nested_ops.transpose_batch_time(inputs)
            states_mask = nested_ops.transpose_batch_time(states_mask)

        if self._state is None:
            with tf.init_scope():
                batch_size = nested_ops.get_size(inputs, dim=1)
                initial_state = self.cell.get_initial_state(
                    inputs=None, batch_size=batch_size, dtype=tf.float32)
                self._state = tf.nest.map_structure(
                    lambda s: tf.Variable(s, trainable=False), initial_state)

        current_state = tf.nest.map_structure(lambda x: tf.identity(x), self._state)

        iters = nested_ops.get_size(inputs, dim=0)

        common_args = (self.cell, inputs, current_state, states_mask)

        if not tf.is_tensor(iters) and iters == 1:
            outputs, last_state = _single_step(*common_args)
        else:
            outputs, last_state = _multi_step(
                *common_args,
                iters=iters,
                parallel_iterations=self._parallel_iterations,
                swap_memory=self._swap_memory)

        # update state
        tf.nest.map_structure(
            lambda dst, src: dst.assign(src), self._state, last_state)

        if not self._time_major:
            outputs = nested_ops.transpose_batch_time(outputs)

        return outputs
    
    def get_config(self) -> dict:
        base_config = super().get_config()
        base_config.update({
            'cell': {
                'class_name': self.cell.__class__.__name__,
                'config': self.cell.get_config()
            },
            'time_major': self._time_major,
            'parallel_iterations': self._parallel_iterations,
            'swap_memory': self._swap_memory,
        })
        return base_config

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        cell = tf.keras.layers.deserialize(
            config.pop('cell'), custom_objects=custom_objects)
        layer = cls(cell, **config)
        return layer

    def compute_output_shape(self, input_shape):
        return self.cell.compute_output_shape(input_shape)
    

def _single_step(cell, inputs, states, states_mask):
    inputs = nested_ops.squeeze_dim(inputs, 0)
    states_mask = nested_ops.squeeze_dim(states_mask, 0)
    states = tf.nest.map_structure(
        lambda x: tf.where(states_mask, x, tf.zeros_like(x)), states)
    outputs, states = cell(inputs, states)
    outputs = nested_ops.expand_dim(outputs, 0)
    return outputs, states

def _multi_step(cell, inputs, states, states_mask, iters, parallel_iterations, swap_memory):

    _input_time_zero = tf.nest.map_structure(lambda x: x[0], inputs)
    _state_time_zero = tf.nest.map_structure(tf.zeros_like, states)

    output_time_zero, _ = cell(_input_time_zero, _state_time_zero)

    inputs = tf.nest.map_structure(
        lambda x: tf.TensorArray(
            dtype=x.dtype, size=iters, element_shape=x.shape[1:]).unstack(x),
        inputs)

    states_mask = tf.TensorArray(
        dtype=states_mask.dtype,
        size=iters,
        element_shape=states_mask.shape[1:],
    ).unstack(states_mask)

    def cond(i, *_):
        return i < iters

    def body(i, states, sequence_outputs):
        
        inputs_i = tf.nest.map_structure(lambda x: x.read(i), inputs)

        states_mask_i = states_mask.read(i)
        
        states = tf.nest.map_structure(
            lambda x: tf.where(states_mask_i, x, tf.zeros_like(x)), states)

        outputs_i, states = cell(inputs_i, states)
        
        sequence_outputs = tf.nest.map_structure(
            lambda dst, src: dst.write(i, src), 
            sequence_outputs, outputs_i)
        return (i + 1, states, sequence_outputs)

    sequence_outputs = tf.nest.map_structure(
        lambda x: tf.TensorArray(
            dtype=x.dtype,
            size=iters,
            element_shape=x.shape
        ),
        output_time_zero
    )

    _, states, sequence_outputs = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(tf.constant(0), states, sequence_outputs),
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        maximum_iterations=None,
    )
    
    sequence_outputs = tf.nest.map_structure(
        lambda x: x.stack(), sequence_outputs)
    
    if isinstance(iters, int):
        iterations_shape = tf.TensorShape([iters])
        tf.nest.map_structure(
            lambda t: t.set_shape(iterations_shape.concatenate(t.shape[1:])),
            sequence_outputs,
        )

    return sequence_outputs, states

