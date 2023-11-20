import tensorflow as tf

from abc import ABC 
from abc import abstractmethod

from reinforceable.timestep import Timestep

from reinforceable.types import Tensor
from reinforceable.types import NestedSpec
from reinforceable.types import NestedTensor 
from reinforceable.types import FlatTimestep
from reinforceable.types import Shape 
from reinforceable.types import DType


class Environment(ABC):

    '''An abstract environment.'''

    def __init__(
        self, 
        action_spec: NestedSpec, 
        output_spec: Timestep,
    ) -> None:
        self._set_action_spec(action_spec)
        self._set_output_spec(output_spec)
        
    @abstractmethod
    def py_step(self, action: NestedTensor) -> FlatTimestep:
        '''A step in the environment.'''

    @abstractmethod
    def py_reset(self, *, auto_reset: bool) -> FlatTimestep:
        '''A reset of the environment.'''

    @abstractmethod
    def py_render(self) -> Tensor:
        '''A rendering step of the environment.'''

    def reset(self, *, auto_reset: bool = True) -> Timestep:
        flat_timestep = tf.numpy_function(
            self._reset, inp=[auto_reset], Tout=self._flat_output_dtype)
        flat_timestep = tf.nest.map_structure(
            tf.ensure_shape, flat_timestep, self._flat_output_shape)
        return tf.nest.pack_sequence_as(self.output_spec, flat_timestep)

    def step(self, action: NestedTensor) -> Timestep:
        tf.nest.assert_same_structure(self.action_spec, action)
        flat_action = tf.nest.flatten(action)
        flat_timestep = tf.numpy_function(
            self._step, inp=flat_action, Tout=self._flat_output_dtype)
        flat_timestep = tf.nest.map_structure(
            tf.ensure_shape, flat_timestep, self._flat_output_shape)
        return tf.nest.pack_sequence_as(self.output_spec, flat_timestep)
    
    def render(
        self, 
        output_shape: Shape = None, 
        output_dtype: DType = tf.uint8
    ) -> Tensor:
        output = tf.numpy_function(
            self._render, inp=[], Tout=output_dtype)
        if output_shape is not None:
            output = tf.ensure_shape(output, output_shape)
        return output
    
    def _reset(self, auto_reset: bool) -> FlatTimestep:
        timestep = self.py_reset(auto_reset=auto_reset)
        return tf.nest.flatten(timestep)

    def _step(self, *flat_action: list[Tensor]) -> FlatTimestep:
        action = tf.nest.pack_sequence_as(self.action_spec, list(flat_action))
        timestep = self.py_step(action)
        return tf.nest.flatten(timestep)
    
    def _render(self) -> Tensor:
        return self.py_render()
    
    def _set_action_spec(self, action_spec: NestedSpec) -> None:
        self.action_spec = action_spec 
        self.action_shape = tf.nest.map_structure(
            lambda x: x.shape, self.action_spec)
        self.action_dtype = tf.nest.map_structure(
            lambda x: x.dtype, self.action_spec)

    def _set_output_spec(self, output_spec: Timestep) -> None:
        assert isinstance(output_spec, Timestep), (
            '`output_spec` of derived environment needs to be a Timestep.')
        self.output_spec = output_spec
        self.output_shape = tf.nest.map_structure(
            lambda x: x.shape, self.output_spec)
        self.output_dtype = tf.nest.map_structure(
            lambda x: x.dtype, self.output_spec)
        self._flat_output_shape = tf.nest.flatten(self.output_shape)
        self._flat_output_dtype = tf.nest.flatten(self.output_dtype)