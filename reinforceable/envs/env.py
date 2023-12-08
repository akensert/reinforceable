import tensorflow as tf

from abc import ABC 
from abc import abstractmethod

from reinforceable.timestep import Timestep

from reinforceable.types import Tensor
from reinforceable.types import NestedSpec
from reinforceable.types import NestedTensor 
from reinforceable.types import FlatTimestep
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
    def py_reset(self, **kwargs) -> FlatTimestep:
        '''A reset of the environment.'''

    @abstractmethod
    def py_render(self) -> Tensor:
        '''A rendering step of the environment.'''

    # @abstractmethod
    # def random_action(self):
    #     '''A random action.'''

    def reset(self, **kwargs) -> Timestep:
        flat_timestep = self._tf_reset(**kwargs)
        flat_timestep = tf.nest.map_structure(
            tf.ensure_shape, flat_timestep, self._flat_output_shape)
        return tf.nest.pack_sequence_as(self.output_spec, flat_timestep)

    def step(self, action: NestedTensor) -> Timestep:
        tf.nest.assert_same_structure(self.action_spec, action)
        flat_timestep = self._tf_step(action)
        flat_timestep = tf.nest.map_structure(
            tf.ensure_shape, flat_timestep, self._flat_output_shape)
        return tf.nest.pack_sequence_as(self.output_spec, flat_timestep)
    
    def render(self, output_dtype: DType = tf.uint8):
        return self._tf_render(output_dtype=output_dtype)
    
    def _tf_reset(self, **kwargs) -> FlatTimestep:
        kwargs_structure = kwargs 
        def _reset(*flat_kwargs) -> FlatTimestep:
            kwargs = tf.nest.pack_sequence_as(kwargs_structure, flat_kwargs)
            return tf.nest.flatten(self.py_reset(**kwargs))
        flat_kwargs = tf.nest.flatten(kwargs)
        return tf.numpy_function(_reset, flat_kwargs, self._flat_output_dtype)

    def _tf_step(self, action: NestedTensor) -> FlatTimestep:
        action_structure = action 
        def _step(*flat_action: Tensor) -> FlatTimestep:
            action = tf.nest.pack_sequence_as(action_structure, flat_action)
            return tf.nest.flatten(self.py_step(action))
        flat_action = tf.nest.flatten(action)
        return tf.numpy_function(_step, flat_action, self._flat_output_dtype)

    def _tf_render(self, output_dtype: DType) -> Tensor:
        def _render():
            return self.py_render()
        return tf.numpy_function(_render, [], output_dtype)
    
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