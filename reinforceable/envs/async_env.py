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


READY = 0
RESULT = 1
EXCEPTION = 2


# TODO: Can KeyboardInterrupt interrupt the agent-environment interaction
#       without breaking the pipes? 

class AsyncEnvironment(Environment):

    '''An asynchronous environment.
    
    Runs multiple environments in parallel asychronously.

    Example:

    >>> env_constructors = [
    ...     lambda: gym.make('LunarLanderContinuous-v2') for _ in range(8)
    ... ]
    >>> tf_async_env = reinforceable.AsyncEnvironment(
    ...     env_constructors,
    ...     input_signature=tf.TensorSpec((8, 2), tf.float32), # optional
    ...     output_signature=reinforceable.Timestep(
    ...         state=tf.TensorSpec((8, 8), tf.float32),
    ...         step_type=tf.TensorSpec((8, 1), tf.int32),
    ...         reward=tf.TensorSpec((8, 1), tf.float32),
    ...         info={}
    ...     )
    ... )

    Args:
        env_constructors:
            A list of uninitialized environments, which will be initialized
            in the worker process. This argument may take an input like this:
            `env_constructors=[lambda: gym.make(id) for _ in range(32)]` to
            run 32 independent environments in parallel.
        output_signature:
            The signature of the output of `step` and `reset` (namely, time 
            steps). The shapes should include batch_size. The `Timestep` output 
            of `step` and `reset` must have the same structure. For instance, 
            if `step` supply a `Timestep` with certain type information (found 
            in `info`), the `reset` also needs to supply a `Timestep` with that 
            type of information. Currently, this is both required by the 
            `Environment` and the `Driver`.
        input_signature:
            Optional signature of the input to `step` (namely, the signature of
            the action). The shape(s) should include batch_size. 
            Default to None.
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
        output_signature: Timestep,
        input_signature: NestedSpec = None,
        context: str = None,
        seeds: list[int] = None
    ) -> None:
        
        super().__init__(input_signature, output_signature)
        
        context = mp.get_context(context)

        processes = []
        parent_pipes = []

        if seeds is None:
            seeds = list(range(len(env_constructors)))

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

    def py_reset(self, *, auto_reset: bool = True) -> Timestep:
        for parent_pipe in self.parent_pipes:
            parent_pipe.send(('reset', auto_reset))
        return self._receive()
    
    def py_step(self, action: NestedTensor) -> Timestep:
        actions = _nested_unbatch(action)
        for parent_pipe, action in zip(self.parent_pipes, actions):
            parent_pipe.send(('step', action))
        return self._receive()
    
    def py_render(self) -> Tensor:
        for parent_pipe in self.parent_pipes:
            parent_pipe.send(('render', None))
        return self._receive()

    def _receive(self) -> Timestep|NestedTensor|Tensor:
        outputs = []
        for parent_pipe in self.parent_pipes:
            message, output = parent_pipe.recv()
            outputs.append(output)
            if message == EXCEPTION:
                raise Exception(output)
        return _nested_batch(outputs)
    
    def close(self):
        for parent_pipe, process in zip(self.parent_pipes, self.processes):
            parent_pipe.send(('close', None))
            parent_pipe.close()
            process.join()


def _nested_batch(
    inputs: list[Timestep|NestedTensor]
) -> Timestep|NestedTensor:
    flat_inputs = [
        tf.nest.flatten(x, expand_composites=True) for x in inputs
    ]
    stacked_inputs = [np.stack(x) for x in zip(*flat_inputs)]
    return tf.nest.pack_sequence_as(
        inputs[0], stacked_inputs, expand_composites=True)

def _nested_unbatch(
    inputs: Timestep|NestedTensor
) -> list[Timestep|NestedTensor]:
    flat_inputs = tf.nest.flatten(inputs, expand_composites=True)
    unstacked_flat_inputs = [list(x) for x in flat_inputs]
    inputs = [
        tf.nest.pack_sequence_as(inputs, x) 
        for x in zip(*unstacked_flat_inputs)]
    return inputs

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
    env.seed(seed)

    parent_pipe.close()
    
    # Let main process know env is ready.
    child_pipe.send((READY, env.__class__.__name__)) 
    
    try:
        while True:

            command, *data = child_pipe.recv()

            if command == 'step':
                action, = data
                if timestep.step_type == 2:
                    if auto_reset:
                        timestep = env.reset()
                    else:
                        pass
                else:
                    timestep = env.step(action)
                child_pipe.send((RESULT, timestep))

            elif command == 'reset':
                auto_reset, = data
                timestep = env.reset()
                child_pipe.send((RESULT, timestep))

            elif command == 'render':
                output = env.render()
                child_pipe.send((RESULT, output))

            elif command == 'close':
                env.close()
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

