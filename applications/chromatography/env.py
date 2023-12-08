import os

import numpy as np

import gymnasium as gym

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

from env_utils import *


class Chromatography(gym.Env):

    '''Chromatographic environment.'''
    
    def __init__(
        self,
        time: float, 
        num_actions: int,
        num_experiments: int,
        num_compounds: tuple[int, int] = (10, 20),
        resolution_target: float = 1.5,
        time_target: float = 20.0, 
        chromatogram_size: int = 8192,
        initial_action: np.ndarray = None,
        reward_info: bool = False,
        seed: int = None,
        render_fn: callable = None,
        save_path: str = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        
        super(Chromatography, self).__init__()
        
        self.mixture_generator = MixtureGenerator(
            *num_compounds, dtype=dtype)
        
        self.time = Time(
            start=0.0, 
            end=time, 
            delta=time/chromatogram_size, 
            dtype=dtype)
        
        self.program = MobilePhaseProgram(
            time=self.time, 
            num_segments=(num_actions-1), 
            dtype=dtype)
        
        self.crf = ChromatographicResponseFunction(
            resolution_target=resolution_target, 
            time_target=time_target, 
            dtype=dtype)
        
        self.num_experiments = num_experiments

        self._reward_info = reward_info

        self._seed = seed
        self._dtype = dtype
        self._save_path = save_path

        if render_fn is None:
            self._render_fn = _plot_separation
        else:
            self._render_fn = render_fn
        
        if initial_action is None:
            # phi_start at 0.3 (-0.4) to make sure all, or the very majority,
            # of compounds elute within the maximum time (usually between 
            # 20-30 minutes).
            self.initial_action = np.linspace(
                -0.4, 1.0, num_actions, dtype=self._dtype)
        else:
            self.initial_action = np.zeros(
                shape=(num_actions,), dtype=self._dtype) 
            self.initial_action += initial_action

        self.observation_space = gym.spaces.Dict(
            {
                'chromatogram': gym.spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=self.time.array.shape, 
                    dtype=self._dtype),
                'phi_target': gym.spaces.Box(
                    low=-1.0,
                    high=1.0, 
                    shape=self.initial_action.shape, 
                    dtype=self._dtype)
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=self.initial_action.shape, 
            dtype=self._dtype)

    def reset(self, **kwargs) -> tuple:

        self.step_counter = 0
        
        if 'seed' in kwargs:
            self._seed = kwargs.pop('seed')
        elif self._seed is not None:
            self._seed += 1
            
        self.mixture = self.mixture_generator.generate(seed=self._seed)
        scaled_action = self.initial_action / 2. + 0.5
       
        self.program.set(scaled_action)
        self.separation = self.program.run(self.mixture, seed=self._seed)
        self.result = self.crf(self.separation)
        chromatogram = self.separation.observation(self._seed)
        observation = {
            'chromatogram': chromatogram, 'phi_target': self.initial_action}

        info = {}
        if self._reward_info:
            info['reward'] = self.result.reward 

        return observation, info

    def step(self, action: np.ndarray) -> tuple:

        self.step_counter += 1

        action = action.astype(self._dtype)

        scaled_action = action / 2. + 0.5
        
        self.program.set(scaled_action)
        self.separation = self.program.run(self.mixture, seed=self._seed)
        self.result = self.crf(self.separation)
        chromatogram = self.separation.observation(self._seed)
        observation = {
            'chromatogram': chromatogram, 'phi_target': action}

        terminal = truncated = (self.step_counter == self.num_experiments)
            
        info = {}
        if self._reward_info:
            info['reward'] = self.result.reward 

        return observation, self.result.reward, terminal, truncated, info

    def render(self) -> np.ndarray:

        # TODO: Not sure what to render. Right now: save separation as plots,
        #       and return the observation (chromatogram) + reward.

        if self._save_path:
            save_path = (
                f'{self._save_path}_'
                f'seed{self._seed}_'
                f'step{self.step_counter}'
                '.png'
            )
            self._render_fn(
                separation=self.separation, 
                program=self.program, 
                result=self.result, 
                save_path=save_path)
        
        return self.separation.observation(), self.result.reward
    

def _plot_separation(
    separation: Separation, 
    program: MobilePhaseProgram,
    result: float,
    xlabel: str = 'Time (min)',
    y1label: str = 'Absorbance',
    y2label: str = '$\phi$',
    y1lim: tuple[float, float] = (-0.10, 1.50),
    y2lim: tuple[float, float] = (-0.05, 1.05),
    save_path: str = None,
) -> None:
        
    fig, ax1 = plt.subplots(1, 1, figsize=(9, 3))

    blue, orange, green = 'C0', 'C1', 'C2'

    x = separation.time.array
    y = separation.observation()
    
    ax1.set_title(
        r'$\bar{{R}}_{{avg}}$ = {:.3f}'.format(np.mean(result.resolutions)), 
        fontsize=10, 
        color=green)
                  
    ax1.plot(x, y, color=blue, zorder=2)
    ax1.set_xlabel(xlabel, fontsize=10)
    ax1.set_ylabel(y1label, color=blue, fontsize=10)
    # ax1.set_ylim(*y1lim)
    
    ax2 = ax1.twinx()

    ax2.plot(x, program.phi, color=orange, zorder=1)
    ax2.set_ylabel(y2label, color=orange, fontsize=12)
    ax2.set_ylim(*y2lim)
    
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    indices = np.argsort(separation.loc)

    locs = separation.loc[indices]

    for index, loc in zip(indices, locs):
        xtext, ytext = loc, y[np.argmin(np.abs(x - loc))]
        ax1.scatter(xtext, ytext + 0.05, s=10)
      #  ax1.text(xtext, min(ytext, y1lim[1]), f'{index}', fontsize=8, va='bottom', ha='center')

    if save_path: 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)

    plt.clf()
    plt.close()