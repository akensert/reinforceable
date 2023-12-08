import tensorflow as tf
import numpy as np
import multiprocessing as mp
import sys
import cloudpickle
import traceback

from absl import logging

from reinforceable.envs.env import Environment
from reinforceable.timestep import Timestep 

from reinforceable.types import GymEnvironment
from reinforceable.types import GymEnvironmentConstructor
from reinforceable.types import NestedSpec
from reinforceable.types import Tensor 
from reinforceable.types import NestedTensor
from reinforceable.types import FlatTimestep


READY = 0
RESULT = 1
EXCEPTION = 2


class AsyncEnvironment(Environment):

    '''An asynchronous environment.
    
    Runs multiple environments in parallel asychronously.

    Example:

    >>> env_constructors = [
    ...     lambda: gym.make('LunarLanderContinuous-v2') for _ in range(8)
    ... ]
    >>> tf_async_env = reinforceable.envs.AsyncEnvironment(env_constructors)
    >>> initial_timestep = tf_async_env.reset()

    Args:
        env_constructors:
            A list of uninitialized environments, which will be initialized
            in the worker process. This argument may take an input like this:
            `env_constructors=[lambda: gym.make(id) for _ in range(32)]` to
            run 32 independent environments in parallel.
        context:
            Optional context, supplied to multiprocessing.get_context. If None
            the default context is used. Default to None.
        seeds:
            Optional seeds to be supplied to each worker environment. If None
            seeds are list(range(len(env_constructors))). Default to None.
    '''

    def __init__(
        self, 
        env_constructors: list[GymEnvironmentConstructor], 
        context: str = None,
        seeds: list[int] = None
    ) -> None:
        
        batch_size = len(env_constructors)

        # Obtain action spec, reset output spec and step output spec
        dummy_env = env_constructors[0]()
        dummy_env.reset()
        action = dummy_env.action_space.sample()
        timestep = _convert_to_timestep(*dummy_env.step(action))
        action_spec = _batched_spec_from_value(action, batch_size)
        output_spec = _batched_spec_from_value(timestep, batch_size)
        dummy_env.close()
        del dummy_env, action, timestep

        super().__init__(action_spec=action_spec, output_spec=output_spec)
        
        context = mp.get_context(context)

        processes = []
        parent_pipes = []

        if seeds is None:
            seeds = list(range(batch_size))

        for index, (env_ctor, seed) in enumerate(zip(env_constructors, seeds)):

            parent_pipe, child_pipe = context.Pipe()

            env_ctor = cloudpickle.dumps(env_ctor)

            process = context.Process(
                target=_worker,
                args=(index, env_ctor, seed, parent_pipe, child_pipe)
            )

            parent_pipes.append(parent_pipe)
            processes.append(process)

            process.daemon = False 
            process.start()

            child_pipe.close()
            _, name = parent_pipe.recv()

        self.processes = processes
        self.parent_pipes = parent_pipes

    def py_reset(self, **kwargs) -> FlatTimestep:
        auto_reset = kwargs.pop('auto_reset', True)
        seeds = kwargs.pop('seed', None)
        if seeds is None:
            for parent_pipe in self.parent_pipes:
                parent_pipe.send(('reset', auto_reset, None))
        else:
            assert seeds.shape and len(seeds) == len(self.parent_pipes), (
                '`seed` needs to contain a seed for each environment.')
            for parent_pipe, seed in zip(self.parent_pipes, list(seeds)):
                parent_pipe.send(('reset', auto_reset, seed))
        return self._receive()
    
    def py_step(self, action: NestedTensor) -> FlatTimestep:
        actions = _nested_unbatch(action)
        for parent_pipe, action in zip(self.parent_pipes, actions):
            parent_pipe.send(('step', action))
        return self._receive()
    
    def py_render(self) -> Tensor:
        for parent_pipe in self.parent_pipes:
            parent_pipe.send(('render', None))
        return self._receive()
    
    def close(self):
        for parent_pipe, process in zip(self.parent_pipes, self.processes):
            parent_pipe.send(('close', None))
            parent_pipe.close()
            process.join()

    def _receive(self) -> Tensor|NestedTensor|FlatTimestep:
        outputs = []
        exceptions = []
        for parent_pipe in self.parent_pipes:
            message, output = parent_pipe.recv()
            outputs.append(output)
            if message == EXCEPTION:
                exceptions.append(output)
        
        if len(exceptions):
            raise Exception(exceptions)
        
        return _nested_batch(outputs)
    

def _convert_to_timestep(*data: NestedTensor|Tensor) -> Timestep:

    if len(data) == 2:
        observation, info = data
        return Timestep(
            state=observation,
            step_type=np.array([0], np.int32),
            reward=np.array([0.0], np.float32),
            info=info)
    
    observation, reward, terminal, truncated, info = data
    return Timestep(
        state=observation,
        step_type=(
            np.array([2], np.int32) if (terminal or truncated) else 
            np.array([1], np.int32)
        ),
        reward=np.array([reward], np.float32),
        info=info)

def _batched_spec_from_value(
    data: NestedTensor|Timestep, 
    batch_size: int
) -> NestedSpec:
    data = tf.nest.map_structure(tf.convert_to_tensor, data)
    spec = tf.nest.map_structure(
        lambda x: tf.TensorSpec((batch_size,) + x.shape, x.dtype), data)
    return spec 
    
def _nested_batch(
    inputs: list[NestedTensor|Timestep]
) -> NestedTensor|Timestep:
    flat_inputs = [
        tf.nest.flatten(x, expand_composites=True) for x in inputs
    ]
    stacked_inputs = [np.stack(x) for x in zip(*flat_inputs)]
    return tf.nest.pack_sequence_as(
        inputs[0], stacked_inputs, expand_composites=True)

def _nested_unbatch(
    inputs: NestedTensor|Timestep
) -> list[NestedTensor|Timestep]:
    flat_inputs = tf.nest.flatten(inputs, expand_composites=True)
    unstacked_flat_inputs = [list(x) for x in flat_inputs]
    inputs = [
        tf.nest.pack_sequence_as(inputs, x) 
        for x in zip(*unstacked_flat_inputs)]
    return inputs

# TODO: unpack timestep (it is no longer a Timestep)
def _worker(
    index: int, 
    env: bytes, 
    seed: int, 
    parent_pipe: mp.connection.Connection, 
    child_pipe: mp.connection.Connection, 
) -> None:

    env: GymEnvironment = cloudpickle.loads(env)()
    
    # TODO: Sufficient?
    np.random.seed(seed)
    if hasattr(env, 'seed'):
        env.seed(seed)

    parent_pipe.close()
    
    # Let main process know env is ready.
    child_pipe.send((READY, env.__class__.__name__)) 
    
    try:
        while True:

            command, *data = child_pipe.recv()

            if command == 'step':
                if timestep.step_type[0] != 2:
                    timestep = _convert_to_timestep(*env.step(data[0]))
                elif auto_reset:
                    timestep = _convert_to_timestep(*env.reset())
      
                child_pipe.send((RESULT, timestep))

            elif command == 'reset':
                auto_reset, seed = data
                reset_output = env.reset(seed=seed)
                timestep = _convert_to_timestep(*reset_output)
                child_pipe.send((RESULT, timestep))

            elif command == 'render':
                output = env.render()
                child_pipe.send((RESULT, output))

            elif command == 'close':
                env.close()
                child_pipe.send((RESULT, None))
                break

    except (KeyboardInterrupt, Exception):
        exc_type, exc_value, exc_traceback = sys.exc_info()
        stacktrace = ''.join(traceback.format_exception(
            exc_type, exc_value, exc_traceback))
        message = f'Error in environment process {index}: {stacktrace}'
        logging.error(message)
        child_pipe.send((EXCEPTION, message))

    finally:
        env.close()

