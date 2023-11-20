import tensorflow as tf
import numpy as np
import gymnasium as gym

from reinforceable.types import GymEnvironment


class NoInfoEnv(gym.Wrapper):

    # TODO: Change name of class? 

    def reset(
        self, 
        **kwargs
    ) -> tuple[np.ndarray, dict]:
        obs, _ = self.env.reset(**kwargs)
        return obs, {}

    def step(
        self, 
        action: np.ndarray|float|int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, rew, term, trunc, _ = self.env.step(action)
        return obs, rew, term, trunc, {}


class EpisodicLifeEnv(gym.Wrapper):
    
    def __init__(
        self, 
        env: GymEnvironment
    ) -> None:
        super().__init__(env)
        self.perform_reset = True

    def reset(
        self, 
        **kwargs
    ) -> tuple[np.ndarray, dict]:
        if self.perform_reset:
            # Normal reset
            state, info = self.env.reset(**kwargs)
            self.n_lives = info['lives']
            self.perform_reset = False
        else:
            # 'Fire' reset
            state, reward, terminal, truncated, info = self.env.step(0)
            # Game over can occur after fire reset.
            if terminal or truncated:
                state, info = self.env.reset(**kwargs)
        return state, info
        
    def step(
        self, 
        action: np.ndarray|float|int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        state, reward, terminal, truncated, info = self.env.step(action)
        # If number of lives has decreased, make it a terminal state
        if self.n_lives > info['lives']:
            self.n_lives -= 1
            terminal = truncated = True
            # If zero lives left, flag for actual reset
            if not self.n_lives:
                self.perform_reset = True
        return state, reward, terminal, truncated, info


class FloatingStateEnv(gym.Wrapper):

    # TODO: Change name of class? 
    
    def __init__(
        self, 
        env: GymEnvironment, 
        nested: bool = False
    ) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low,
            high=self.observation_space.high,
            shape=self.observation_space.shape,
            dtype=np.float32
        )
        self._nested = nested

    def reset(
        self, 
        **kwargs
    ) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        obs = self._cast_to_float(obs)
        return obs, info 
    
    def step(
        self, 
        action: np.ndarray|float|int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, rew, term, trunc, info = self.env.step(action)
        obs = self._cast_to_float(obs)
        return obs, rew, term, trunc, info

    def _cast_to_float(self, obs):
        if not self._nested:
            return obs.astype(np.float32)
        return tf.nest.map_structure(lambda o: o.astype(np.float32), obs)