import tensorflow as tf

from abc import ABC
from abc import abstractmethod

from reinforceable.timestep import Timestep

from reinforceable.types import Tensor
from reinforceable.types import DType


class Observer(tf.Module, ABC):

    '''An abstract observer.'''

    def __init__(self, name: str = None) -> None:
        super().__init__(name=name)
        self.built = False 

    @abstractmethod
    def call(self, timestep: Timestep) -> None:
        pass 

    @abstractmethod
    def build(self, timestep: Timestep) -> None:
        pass 

    @abstractmethod
    def result(self) -> Tensor:
        pass

    def __call__(self, timestep: Timestep) -> None:
        
        if not self.built:
            with self.name_scope:
                self.build(timestep)
                self.built = True

        self.call(timestep)


class Counter(Observer):

    '''An abstract counter observer. 
    
    Implemented for internal use.
    
    '''

    def __init__(
        self, 
        reset_on_result: bool = False, 
        dtype: DType = tf.int64,
        name: str = None
    ) -> None:
        super().__init__(name=name)
        self._reset_on_result = reset_on_result
        self._dtype = dtype

    def call(self, timestep: Timestep) -> None:
        self.counter.assign_add(tf.squeeze(self._call(timestep), -1))

    def build(self, timestep: Timestep) -> None:
        batch_size = timestep.step_type.shape[0]
        self.counter = tf.Variable(
            initial_value=tf.zeros((batch_size,), dtype=self._dtype), 
            dtype=self._dtype,
            trainable=False)
    
    def result(self) -> tf.Tensor:
        result = tf.reduce_sum(self.counter)
        if self._reset_on_result:
            self.reset()
        return result 
    
    def reset(self) -> None:
        tf.nest.map_structure(
            lambda x: x.assign(tf.zeros_like(x)), self.variables)
        
    @abstractmethod
    def _call(self, timestep: Timestep) -> tf.Tensor:
        pass
    
    
class EpisodicRollingAverage(Observer):

    '''An abstract episodic rolling average observer.
    
    Implemented for internal use.

    '''

    def __init__(
        self, 
        window_size: int, 
        reset_on_result: bool = False,
        dtype: DType = tf.float32, 
        name: str = None
    ):
        super().__init__(name=name)
        self._window_size = window_size
        self._reset_on_result = reset_on_result
        self._dtype = dtype

    def call(self, timestep: Timestep) -> None:
        
        # TODO: Clean up.

        self._episode_value.assign_add(tf.squeeze(self._call(timestep), -1))

        terminal = tf.squeeze(timestep.terminal(), -1)
        terminal_float = tf.cast(terminal, self._dtype)
        terminal_int = tf.cast(terminal, tf.int64)

        self._divisor.assign_add(terminal_float)
        self._divisor.assign(
            tf.minimum(
                tf.cast(self._window_size, self._divisor.dtype), 
                self._divisor
            )
        )

        # Comments for debugging
        # Assume terminal = [True, False, False, True] 
        terminal_indices = tf.squeeze(tf.where(terminal), -1)
        # -> [0, 3]

        # assume pointer = [6, 7, 1, 2]
        pointer = tf.gather(self._pointer, terminal_indices)
        # -> [6, 2]

        # assume value is [3., 6., 1., 2.]
        value = tf.gather(self._episode_value, terminal_indices)
        # -> [3., 2.]

        pointer = tf.stack([pointer, terminal_indices], axis=1)
        # -> [[6, 0], [2, 3]]

        self._episode_value_list.scatter_nd_update(pointer, value)
        # update [[6, 0], [2, 3]] with values [3., 2.]

        self._episode_value.assign(self._episode_value * (1.0 - terminal_float))
        
        self._pointer.assign(
            tf.math.mod(
                self._pointer + terminal_int, 
                self._window_size
            )
        )
        
    def build(self, timestep: Timestep) -> None:
        batch_size = timestep.reward.shape[0]
        window_size = self._window_size
        self._episode_value = tf.Variable(
            initial_value=tf.zeros((batch_size,), self._dtype), 
            dtype=self._dtype,
            trainable=False)
        self._episode_value_list = tf.Variable(
            initial_value=tf.zeros((window_size, batch_size), self._dtype), 
            dtype=self._dtype,
            trainable=False)
        self._pointer = tf.Variable(
            initial_value=tf.zeros((batch_size,), tf.int64), 
            dtype=tf.int64,
            trainable=False)
        self._divisor = tf.Variable(
            initial_value=tf.zeros((batch_size,), self._dtype), 
            dtype=self._dtype,
            trainable=False)

    def result(self) -> tf.Tensor:
        # TODO: Clean up.
        result = tf.reduce_sum(
            tf.math.divide_no_nan(
                tf.reduce_sum(self._episode_value_list, axis=0), 
                self._divisor
            )
        ) / tf.maximum(1.0, tf.reduce_sum(tf.where(self._divisor > 0, 1.0, 0.0)))
        if self._reset_on_result:
            self.reset()
        return result
    
    def reset(self) -> None:
        tf.nest.map_structure(
            lambda x: x.assign(tf.zeros_like(x)), self.variables)
    
    @abstractmethod
    def _call(self, timestep: Timestep) -> tf.Tensor:
        pass


class StepCounter(Counter):
    def _call(self, timestep: Timestep) -> tf.Tensor:
        return tf.cast(tf.ones_like(timestep.step_type), self._dtype)
    

class EpisodeCounter(Counter):
    def _call(self, timestep: Timestep) -> None:
        return timestep.terminal(self._dtype)
    

class RollingAverageEpisodeLength(EpisodicRollingAverage):
    def _call(self, timestep: Timestep) -> tf.Tensor:
        return tf.ones_like(timestep.reward)
    

class RollingAverageEpisodeReturn(EpisodicRollingAverage):
    def _call(self, timestep: Timestep) -> tf.Tensor:
        return timestep.reward 
    

