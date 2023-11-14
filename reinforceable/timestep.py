import tensorflow as tf

import numpy as np

from typing import NamedTuple

from reinforceable.types import TensorOrSpec
from reinforceable.types import NestedTensorOrSpec
from reinforceable.types import Shape
from reinforceable.types import DType
from reinforceable.types import Tensor 


class Timestep(NamedTuple):

    '''The time step of an environment. 
    
    The time step is supplied by step and reset methods of the environment.
    
    Args:
        state:
            The state of the environment.
        step_type:
            The step type of the environment. Either 0, 1 or 2, where
            0 indicate initial state, 1 intermediate state, and 2 final state,
            respectively.
        reward:
            The reward supplied by the environment.
        info:
            Auxiliary information of the environment. Auxiliary information
            may also include information produced by the agent, such as the
            action, action log probability, or state value.
    '''
    
    state: NestedTensorOrSpec
    step_type: TensorOrSpec
    reward: TensorOrSpec
    info: dict[str, NestedTensorOrSpec]
    
    def terminal(self, dtype: DType = tf.bool) -> Tensor:
        terminal = (self.step_type == 2)
        if dtype == tf.bool:
            return terminal
        return tf.cast(terminal, dtype)
    
    def discount(self, discount_value: float, /) -> Tensor:
        if not tf.is_tensor(discount_value):
            discount_value = tf.convert_to_tensor(discount_value)
        return discount_value * (1.0 - self.terminal(discount_value.dtype))

    @classmethod
    def from_shape(cls, shape: Shape) -> 'Timestep':
        shape = tf.TensorShape(shape)
        batch_shape = shape[:1]
        return cls(
            state=tf.TensorSpec(shape, tf.float32),
            step_type=tf.TensorSpec(
                batch_shape.concatenate([1]), tf.int32),
            reward=tf.TensorSpec(
                batch_shape.concatenate([1]), tf.float32),
            info={}
        )
    
    def __getattr__(self, name: str) -> NestedTensorOrSpec:
        if name in self.info:
            return self.info[name]
        elif name == 'action_mask':
            return None 
        raise AttributeError(f'{name!r} not found.')

    @staticmethod
    def convert_to_timestep_fn(self, *components):
        if len(components) > 2:
            state, reward, terminal, truncated, info = components 
            step_type = [2] if (terminal or truncated) else [1]
        else:
            state, info = components 
            reward = 0.0
            step_type = 0
        return self.__class__(
            state=state,
            reward=np.expand_dims(
                np.asarray(reward, dtype=np.float32), axis=1),
            step_type=np.expand_dims(
                np.asarray(step_type, dtype=np.int32), axis=1),
            info=info
        )