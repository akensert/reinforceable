import tensorflow as tf

from reinforceable.agents.agent import Agent 
from reinforceable.buffer import Buffer
from reinforceable.envs.env import Environment
from reinforceable.utils.observers import Observer

from reinforceable.utils import nested_ops 

from reinforceable.timestep import Timestep
from reinforceable.trajectory import Trajectory


class Driver(tf.Module):

    '''The driver of the agent-environment interaction.
    
    Args:
        agent:
            The agent.
        environment:
            The environment.
        observers:
            A nested structure of observers (e.g., a dict). The observers
            observes the timestep at each step, accumulating information
            such as episode length or episode return. 
        driver_eagerly:
            Whether to run the agent-environment interaction eagerly. Default
            to False.
        name:
            Name of the driver, module.
    '''

    MAX_STEPS = 100_000
    MAX_EPISODES = 1_000

    def __init__(
        self, 
        agent: Agent, 
        env: Environment, 
        observers: dict[str, Observer] = None,
        drive_eagerly: bool = False,
        name: str = 'Driver'
    ):
        super().__init__(name=name)
        self.agent = agent 
        self.env = env
        self.observers = observers
        self.drive_eagerly = drive_eagerly
        self.current_timestep = None
        if not self.drive_eagerly:
            self._interaction_loop = tf.function(self._interaction_loop) 
            
    def run(
        self, 
        steps: int = 128, 
        episodes: int = None, 
        **kwargs
    ) -> Trajectory:
        
        '''Runs the driver. 
        
        In other words, runs the interaction between the agent and environment.
        This method is pretty much just a wrapper around `_interaction_loop`
        which is by default wrapped in a `tf.function` for speed.

        Args:
            steps:
                Number of steps of interaction. Default to 128.
            episodes:
                Number of episodes of interaction. Either `steps` or `episodes`
                needs to be specified. If both are specified, the interaction 
                will terminate when `steps` steps and `episodes` episodes have 
                been performed. Default to None.
            **kwargs:
                Any keyword arguments passed the `__call__` method of the 
                agent. Commonly, arguments such as `deterministic` or 
                `training`. 
       
        Returns:
            Trajectory data, sampled from the `Buffer`. For PPO, this data 
            can directly be used to compute advantages and returns, and 
            subsquently used to update the PPO agent. For DDPG, TD3 or SAC, 
            this data can be added to the experience replay buffer, which can 
            then subsquently be sampled to train the agent.
        '''

        if not steps and not episodes:
            raise ValueError(
                'Found both `steps` and `episodes` to be `None`.'
                'Please specify at least one of them.')
        elif steps is None:
            steps = self.MAX_STEPS 
        elif episodes is None:
            episodes = self.MAX_EPISODES

        trajectory, last_timestep = self._interaction_loop(
            self.current_timestep, steps, episodes, **kwargs)

        self.current_timestep = last_timestep

        return self.agent.finalize_trajectory(trajectory, last_timestep)
    
    def result(self):

        '''Returns the result of the observers.'''

        if self.observers is None:
            return None 
        return tf.nest.map_structure(lambda obs: obs.result(), self.observers)
    
    def _interaction_loop(
        self,
        initial_timestep: Timestep,
        steps: int,
        episodes: int,
        **kwargs
    ) -> tuple[Trajectory, Timestep]:
        
        '''The interaction loop.'''

        def _interaction_cond(
            step_counter: tf.Tensor, 
            episode_counter: tf.Tensor, 
            *_
        ) -> bool:
            
            '''The interaction condition. Condition of tf.while_loop.'''

            return tf.logical_and(
                tf.less(tf.reduce_sum(step_counter), steps),
                tf.less(tf.reduce_sum(episode_counter), episodes))
        
        def _interaction_step_and_collect(
            step_counter: tf.Tensor, 
            episode_counter: tf.Tensor, 
            timestep: Timestep, 
            buffer: Buffer
        ) -> tuple[tf.Tensor, tf.Tensor, Timestep, Buffer]:

            '''The interaction step, and collect. Body of tf.while_loop.'''

            timestep, next_timestep = _interaction_step(timestep, **kwargs)

            if self.observers is not None:    
                tf.nest.map_structure(lambda obs: obs(timestep), self.observers)

            buffer = buffer.add(timestep)
            
            step_counter += tf.ones_like(step_counter)  
            episode_counter += timestep.terminal(episode_counter.dtype)

            return (step_counter, episode_counter, next_timestep, buffer)
        
        def _interaction_step(timestep: Timestep) -> tuple[Timestep, Timestep]:
            
            '''The interaction step, invoked in _interaction_step_and_collect.'''

            timestep = nested_ops.expand_dim(timestep, 0)
            
            action, info = self.agent(timestep, **kwargs)

            action = nested_ops.squeeze_dim(action, 0)
            info = nested_ops.squeeze_dim(info, 0)
            timestep = nested_ops.squeeze_dim(timestep, 0)
        
            next_timestep = self.env.step(action)

            info['action'] = action

            timestep.info.update(info)

            return timestep, next_timestep 
        
        if initial_timestep is None:
            initial_timestep = self.env.reset()

        # Perform first interaction step outside loop to obtain transition
        timestep, next_timestep = _interaction_step(initial_timestep, **kwargs)

        # Now initialize buffer (and immediately add this first timestep).
        buffer = Buffer.initialize_add(timestep)

        if self.observers is not None:                
            tf.nest.map_structure(lambda obs: obs(timestep), self.observers)

        step_counter = tf.ones_like(timestep.step_type)  
        episode_counter = timestep.terminal(step_counter.dtype)

        (_, _, next_timestep, buffer) = tf.while_loop(
            cond=_interaction_cond,
            body=_interaction_step_and_collect,
            loop_vars=(step_counter, episode_counter, next_timestep, buffer),
            parallel_iterations=1)

        data = buffer.sample()

        return data, next_timestep