import tensorflow as tf
import numpy as np

from keras.callbacks import Callback
from keras.callbacks import CallbackList

from abc import ABC
from abc import abstractmethod

from reinforceable.envs.env import Environment
from reinforceable.timestep import Timestep
from reinforceable.trajectory import Trajectory

from reinforceable.types import Tensor
from reinforceable.types import NestedTensor
from reinforceable.types import TrainInfo

from reinforceable.utils import nested_ops


class Agent(tf.Module, ABC):

    '''An abstract agent.'''
    
    def __init__(
        self, 
        train_callbacks: list[Callback], 
        train_eagerly: bool = False,
        summary_writer: str|tf.summary.SummaryWriter = None, 
        name: str = None
    ) -> None:
        super().__init__(name=name)
        self.callback = CallbackList(train_callbacks, model=self)
        self.train_eagerly = train_eagerly

        if isinstance(summary_writer, str):
            self.summary_writer = tf.summary.create_file_writer(summary_writer)
        else:
            self.summary_writer = summary_writer
        
        with self.name_scope:
            self.train_counter = tf.Variable(
                initial_value=0, dtype=tf.int64, trainable=False)
            self.epoch_counter = tf.Variable(
                initial_value=0, dtype=tf.int64, trainable=False)
            self.batch_counter = tf.Variable(
                initial_value=0, dtype=tf.int64, trainable=False)

        self.train_step = (
            self.train_step if self.train_eagerly else 
            tf.function(self.train_step))
        
    @abstractmethod
    def __call__(
        self, 
        timestep: Timestep, 
        **kwargs
    ) -> tuple[NestedTensor, dict[str, NestedTensor]]:
        
        '''Computes an action based on a timestep.
        
        This method needs to be implemented for the agent-environment 
        interaction, invoked by the `Driver`.

        Args:
            timestep:
                The current timestep of the environment. Has four fields: 
                `state`, `step_type`, `reward` and `info`.
            **kwargs:
                Any auxiliary information that should be added to the 
                trajectory buffer. For instance, action log probabilities
                and state values.

        Returns:
            The action and optional auxiliary information.
        '''

    @abstractmethod
    def train_step(
        self, 
        data: Trajectory,
        sample_weight: Tensor = None,
    ) -> TrainInfo:
        
        '''A training step on a batch of data.
        
        This method is necessary for computing losses and subsquently updating
        the learnable weights of the agent. Invoked by `self.train(...)`.
        
        Args:
            data:
                Batchable trajectory of data (stack of time steps) to be 
                trained on.
            sample_weight:
                Optional sample weights. Default to None.
        
        Returns:
            Training info, such as losses.
        '''

    def finalize_trajectory(
        self, 
        trajectory: Trajectory, 
        last_timestep: Timestep
    ) -> Trajectory:
        
        '''Preprocessing of trajectory data.
        
        Called at the end the agent-environment interaction (`driver.run(...)`) 
        to process the trajectory data obtained from the buffer. The processed 
        trajectory data will be used to train the agent. 

        Args:
            trajectory: 
                Trajectory data obtained from the buffer. 
            last_timestep:
                The last timestep of the agent-environment interaction. 
                This timestep was not added to the current buffer, but will be 
                added as the first timestep in the next buffer.

        Returns:
            A processed batchable trajectory. By default, this method does 
            not perform any preprocessing.
        '''

        return trajectory

    def train(
        self,
        data: Trajectory, 
        sample_weight: Tensor = None,
        batch_size: int = 32, 
        repeats: int = 10,
    ) -> TrainInfo:
        
        '''Training of the agent.

        Based on collected data from agent-environment interactions, 
        processed by the `finalize_trajectory` method, the agent invokes its 
        `train_step` to compute losses and subsquently update its trainable 
        weights.

        Args:
            data:
                The collected data to train on.
            sample_weight:
                Optional sample weights. Default to None.
            batch_size:
                For trajectory data (T, B, ...), `batch_size` defines the 
                batch length, wherein the time dimension is chunked up as 
                follows: (T[i * batch_size: (i + 1) * batch_size], B, ...]).
                Default to 32.
            repeats:
                The number of complete training passes of the data. 
                Default to 10.
        
        Returns:
            Any information obtained during the training.
        '''

        if sample_weight is not None:
            data = (data, sample_weight)
        else:
            data = (data,)

        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.batch(
            batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.enumerate()
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        self.callback.on_train_begin()

        for epoch in range(repeats):
            
            self.callback.on_epoch_begin(epoch)

            for batch, x in dataset:
                self.callback.on_batch_begin(batch)

                if self.summary_writer is not None:
                    with self.summary_writer.as_default(self.batch_counter):
                        batch_info = self.train_step(*x)
                else:
                    batch_info = self.train_step(*x)

                if batch == 0:
                   epoch_info = batch_info
                else:
                   epoch_info = tf.nest.map_structure(
                       lambda x, y: x + y, epoch_info, batch_info)

                self.callback.on_batch_end(batch, batch_info)

                self.batch_counter.assign_add(1)

            self.callback.on_epoch_end(epoch, epoch_info)

            if epoch == 0:
                train_info = epoch_info
            else:
                train_info = tf.nest.map_structure(
                    lambda x, y: x + y, train_info, epoch_info)

            self.epoch_counter.assign_add(1)

        train_info = tf.nest.map_structure(lambda x: x / repeats, train_info)
        self.callback.on_train_end(train_info)

        self.train_counter.assign_add(1)

        return train_info
    
    def save(self, path: str, *args, **kwargs):
        original_call = self.__call__
        self.__call__ = tf.function(self.__call__)
        self.__call__.get_concrete_function(*args, **kwargs)
        tf.saved_model.save(self, path)
        self.__call__ = original_call

    def _play(
        self, 
        env: Environment, 
        pad: int = 0,
        seed: list[int] = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        
        '''Interacts with the environment for a single episode.
        
        Note: currently implemented for debugging, with an "ugly" padding 
        argument (because of the stateful RNN). Should be modified in the 
        future and made public.
        '''

        timestep = env.reset(auto_reset=False, seed=seed)
        episode_reward = timestep.reward    # [0.0, 0.0, ..., 0.0]
        episode_length = timestep.step_type # [0, 0, ..., 0]
        
        while not all(timestep.terminal()):

            if pad > 0:
                timestep = tf.nest.map_structure(
                    lambda x: tf.pad(x, [(0, pad)] + [(0, 0)] * (len(x.shape)-1) ), timestep)
            
            timestep = nested_ops.expand_dim(timestep, 0)
            
            action, _ = self(timestep, **kwargs)
            action = nested_ops.squeeze_dim(action, 0)
            
            if pad > 0:
                action = action[:-pad]

            next_timestep = env.step(action)

            episode_reward += tf.where(
                timestep.terminal(), 0.0, next_timestep.reward)
            
            episode_length += tf.where(
                timestep.terminal(), 0, 1)
            
            timestep = next_timestep

            env.render()

        return episode_reward.numpy(), episode_length.numpy()
