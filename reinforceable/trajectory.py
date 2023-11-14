import tensorflow as tf

from reinforceable.timestep import Timestep


class Trajectory(Timestep):

    '''A trajectory of time steps.
    
    As of now, a `Trajectory` has the same structure and implementation as a 
    `Timestep` though encoding a stack of time steps. For instance, while a 
    `Timestep` will have shapes (B, ...), a `Trajectory` will have shapes (T, B, ...).

    Args:
        state:
            The states of the environment.
        step_type:
            The step types of the environment. Either 0, 1 or 2, where
            0 indicate initial state, 1 intermediate state, and 2 final state,
            respectively.
        reward:
            The rewards supplied by the environment.
        info:
            Auxiliary information of the environment. Auxiliary information
            may also include information produced by the agent, such as
            actions, action log probabilities, or state values.
    '''
