import tensorflow as tf

from abc import ABC 
from abc import abstractmethod

from reinforceable.types import Tensor
from reinforceable.types import NestedSpec
from reinforceable.types import NestedTensor 
from reinforceable.types import FlatTimestep

from reinforceable.timestep import Timestep


class Environment(ABC):

    '''An abstract environment.'''

    def __init__(
        self, 
        input_signature: NestedSpec, 
        output_signature: Timestep,
    ) -> None:
        self._set_input_spec(input_signature)
        self._set_output_spec(output_signature)
        
    @abstractmethod
    def py_step(self, action: NestedTensor) -> Timestep:
        '''A step in the environment.'''

    @abstractmethod
    def py_reset(self, *, auto_reset: bool) -> Timestep:
        '''A reset of the environment.'''

    @abstractmethod
    def py_render(self) -> Tensor:
        '''A rendering step of the environment.'''

    def reset(self, *, auto_reset: bool = True) -> Timestep:
        flat_timestep = tf.numpy_function(
            self._reset, inp=[auto_reset], Tout=self._flat_output_dtypes)
        flat_timestep = tf.nest.map_structure(
            lambda x, shape: tf.ensure_shape(x, shape), 
            flat_timestep, self._flat_output_shapes)
        timestep = tf.nest.pack_sequence_as(
            self.output_spec, flat_timestep)
        return timestep

    def step(self, action: NestedTensor) -> Timestep:
        if self.input_spec is None:
            # with tf.init_scope(): ??
            self._set_input_spec(action)
        tf.nest.assert_same_structure(self.input_spec, action)
        flat_action = tf.nest.flatten(action)
        flat_timestep = tf.numpy_function(
            self._step, inp=flat_action, Tout=self._flat_output_dtypes)
        flat_timestep = tf.nest.map_structure(
            lambda x, shape: tf.ensure_shape(x, shape), 
            flat_timestep, self._flat_output_shapes)
        timestep = tf.nest.pack_sequence_as(
            self.output_spec, flat_timestep)
        return timestep
    
    def render(self, output_shape=None, output_dtype=tf.uint8) -> Tensor:
        output = tf.numpy_function(
            self._render, inp=[], Tout=output_dtype)
        if output_shape is not None:
            output = tf.ensure_shape(output, output_shape)
        return output
    
    def _reset(self, auto_reset: bool) -> FlatTimestep:
        flat_timestep = tf.nest.flatten(self.py_reset(auto_reset=auto_reset))
        return flat_timestep

    def _step(self, *flat_action: list[Tensor]) -> FlatTimestep:
        action = tf.nest.pack_sequence_as(self.input_spec, list(flat_action))
        flat_timestep = tf.nest.flatten(self.py_step(action))
        return flat_timestep
    
    def _render(self) -> Tensor:
        return self.py_render()
    
    def _set_input_spec(self, input_signature):
        if input_signature is not None:
            self.input_spec = tf.nest.map_structure(
                lambda x: tf.TensorSpec(x.shape, x.dtype), input_signature)
            self.input_dtypes = tf.nest.map_structure(
                lambda x: x.dtype, self.input_spec)
            self.input_shapes = tf.nest.map_structure(
                lambda x: x.shape, self.input_spec)
        else:
            self.input_spec = None
            self.input_dtypes = None
            self.input_shapes = None
            
    def _set_output_spec(self, output_signature):
        self.output_spec = tf.nest.map_structure(
            lambda x: tf.TensorSpec(x.shape, x.dtype), output_signature)
        self.output_dtypes = tf.nest.map_structure(
            lambda x: x.dtype, self.output_spec)
        self.output_shapes = tf.nest.map_structure(
            lambda x: x.shape, self.output_spec)
        self._flat_output_dtypes = tf.nest.flatten(self.output_dtypes)
        self._flat_output_shapes = tf.nest.flatten(self.output_shapes)