import tensorflow as tf

from typing import NamedTuple

from reinforceable.timestep import Timestep
from reinforceable.trajectory import Trajectory

from reinforceable.types import TensorArray
from reinforceable.types import NestedTensorArray


class Buffer(NamedTuple):

    '''The trajectory buffer of the driver.
    
    Utilized by the `driver.Driver` to temporarily store transitions, and then 
    later returns the resulting trajectory for the agent to train on.
    
    Args:
        state: 
            The current state of the environment
        step_type:
            The current step type of the environment.
        reward:
            The current reward of the environemnt.
        info:
            Auxiliary information, supplied by the environment (along with the 
            state, step_type and reward) and/or the agent (via `__call__()`).
    '''

    state: NestedTensorArray
    step_type: TensorArray
    reward: TensorArray
    info: dict[str, NestedTensorArray]

    @classmethod
    def initialize_add(cls, timestep: Timestep) -> 'Buffer':
        timestep_arrays = tf.nest.map_structure(
            lambda x: tf.TensorArray(
                x.dtype, 
                size=0, 
                dynamic_size=True, 
                element_shape=(None,) + x.shape[1:]),
            timestep)
        buffer = cls(
            state=timestep_arrays.state,
            step_type=timestep_arrays.step_type,
            reward=timestep_arrays.reward,
            info=timestep_arrays.info)
        return buffer.add(timestep)

    def add(self, timestep: Timestep) -> 'Buffer':
        i = self.reward.size()
        return tf.nest.map_structure(
            lambda arr, x: arr.write(i, x), self, timestep, check_types=False)

    def sample(self) -> Trajectory:
        trajectory_data = tf.nest.map_structure(lambda arr: arr.stack(), self)
        return Trajectory(
            state=trajectory_data.state,
            step_type=trajectory_data.step_type,
            reward=trajectory_data.reward,
            info=trajectory_data.info)
    