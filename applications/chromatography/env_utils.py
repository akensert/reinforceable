import numpy as np

from typing import NamedTuple 

from math import ceil
from math import floor 


class MixtureGenerator:

    '''Generation of random mixtures.
    
    Specifically, Neue-Kuss parameters kw, s1 and s2 are randomly generated,
    and serves the compounds of the mixture. A given compound is defined as
    a given set of Neue-Kuss parameters. The only thing known about the 
    compound is its behavior (namely, kw, s1 and s2).

    Args:
        min_compounds:
            The minimum number of compounds (sets of Neue-Kuss parameters)
            to generate.
        max_compounds:
            The maximum number of compounds to generate.
        dtype:
            The dtype of the output.
    '''

    def __init__(
        self, 
        min_compounds: int, 
        max_compounds: int, 
        dtype: np.dtype = np.float32
    ):
        self.min_compounds = min_compounds
        self.max_compounds = max_compounds
        self.dtype = dtype
        
    def generate(
        self, 
        num_compounds: int = None, 
        seed: int = None
    ) -> np.ndarray:
        
        '''Generates the mixture.
        
        Args:
            num_compounds:
                Can optionally pass num_compounds which "overwrites" 
                `self.min_compounds` and `self.max_compounds`. Useful if a
                mixture of specific size should be generated.
            seed:
                The random seed passed for reproducability.
        
        Returns:
            The mixture, a stack of Neue-Kuss parameters (kw, s1, s2).
        '''

        rng = np.random.RandomState(seed)

        n = num_compounds or rng.randint(
            self.min_compounds, self.max_compounds + 1)
        
        if n == 0:
            raise ValueError('Cannot generate 0 compounds.')
        
        # Formulas obtained based on behaviors of real compounds:
        random_s1 = lambda: (
            rng.uniform(10.0, 50.0, size=[n])
        )
        random_s2 = lambda s1: (
            (np.log10(s1) * 2.501 - 2.082) + rng.uniform(-0.35, 0.35, size=[n])
        )
        random_kw = lambda s1: (
            10**((s1 * 0.08391 + 0.50544) + rng.uniform(-1.20, 1.20, size=[n]))
        )
    
        s1 = random_s1()
        s2 = random_s2(s1)        
        kw = random_kw(s1)
    
        return np.stack([kw, s1, s2], axis=1).astype(self.dtype)
    

class Time:

    '''The time of a chromatographic run.

    Could just be an array, but for convenience is implemented as a custom
    class: attributes such as `self.end`, `self.delta` and `self.size` will 
    be accessed later on.

    Args:
        start:
            Determines the starting time of the run, usually 0.0 (minutes).
        end:
            Specifies the ending time of the run, where it strictly stops.
        delta:
            Specifies the time step, t_{i+1} - t_{i}. A small value will 
            increase the resolution (number of data points) of the output 
            (e.g., chromatogram). A large value will do the opposite.
            E.g., a delta of 0.001, would result in 10k data points.
        dtype:
            The dtype of the array.
    '''

    def __init__(
        self, 
        start: float, 
        end: float, 
        delta: float, 
        dtype: np.dtype = np.float32
    ):
        self.start = start 
        self.end = end 
        self.delta = delta 
        self.dtype = dtype
        self._size = floor(self.end / self.delta)

    @property
    def array(self):
        time = np.linspace(self.start, self.end, self.size)
        return np.expand_dims(time, axis=1).astype(self.dtype)

    @property
    def size(self):
        return self._size
    

class MobilePhaseProgram:

    '''The mobile phase program.

    The mobile phase program is programmable and is based on the actions 
    of the agent. When run, it produces a `Separation`.

    Args:
        time: 
            The time (`Time`). 
        num_segments:
            The number of segments. In other words, the number 
            of linear gradients, concatenated. If num_segments=30,
            then the agent has to select 30 linear gradients (specifically,
            31 phi targets, as it has to select the initial phi).
        dtype:
            The dtype of the output.
    '''
    
    PHI_MIN: float = 0.0
    PHI_MAX: float = 1.0
    VOID_TIME: float = 1.0
    
    def __init__(
        self, 
        time: Time, 
        num_segments: int, 
        dtype: np.dtype = np.float32
    ):
        self.time = time
        self.num_segments = num_segments
        self.dtype = dtype
        self.phi = None

    def set(self, action: np.ndarray) -> None:

        '''Sets the mobile phase program.

        In other words, specifies how phi should be programmed,
        based on the action of the agent.

        Args:
            action:
                The action of the agent.
        '''
        
        assert len(action) == (self.num_segments + 1), (
            f'num actions ({len(action)}) != '
            f'num segments + 1 ({self.num_segments + 1}).'
        )
        
        steps = ceil(self.time.size / self.num_segments)
        phi_target = np.clip(action, self.PHI_MIN, self.PHI_MAX)
        phi = np.linspace(phi_target[:-1], phi_target[1:], steps)
        self.phi = np.reshape(np.transpose(phi), [-1, 1])[:self.time.size]
        # assert self.phi.shape == self.time.array.shape

    def run(self, compounds: np.ndarray, seed: int = None) -> 'Separation':

        '''Runs the compounds with the program.

        In other words, performs the specified chromatographic run.

        This function basically runs numerical Neue-Kuss retention models
        based on phi (obtained from self.set(action)) and the Neue-Kuss 
        parameters (compounds).

        Note: The stationary phase and other chromatographic factors are 
        encoded in the compounds, each of which is a set of Neue-Kuss parameters.

        Args:
            compounds:
                Each row encodes a compound (via three parameters).
                These three parameters are the parameters of a retention 
                model, which determines behavior of the compound. So a 
                compound is defined as an entity with a single property: 
                a certain behavior based on phi.
            seed:
                The random seed for reproducability.

        Returns:
            A separation.

        Reference:
            https://pubs.acs.org/doi/pdf/10.1021/ac0506783
        '''

        rng = np.random.RandomState(seed)
        
        kw, s1, s2 = compounds.T
        phi = self.phi
        time_delta = self.time.delta 
        void_time = self.VOID_TIME

        k = kw * (1 + s2 * phi)**2 * np.exp(-(s1 * phi) / (1 + s2 * phi))
        # assert k.shape == phi.shape[:-1] + kw.shape
        time_in_column = np.cumsum((1 + k) / k * time_delta, axis=0)
        distance_travelled = np.cumsum(time_delta / (void_time * k), axis=0)
        retention_times = np.min(
            np.where(
                distance_travelled >= 1.0, 
                time_in_column, 
                float('inf')
            ), 
            axis=0
        )

        amplitude = rng.uniform(low=0.5, high=1.0, size=retention_times.shape)
        scale = rng.uniform(low=0.020, high=0.025, size=retention_times.shape)

        return Separation(
            time=self.time, 
            loc=retention_times, 
            scale=scale,
            amplitude=amplitude,
            dtype=self.dtype)


class Separation:

    '''The separation, resulting from `MobilePhaseProgram.run(compounds)`.

    The separation represents the state of a chromatographic environment.

    Args:
        x:
            The time array (obtained from `Time.as_array()`).
        loc:
            The location of the compounds (the peak locations).
        scale:
            The dispersion of the compounds (the width of the peaks).
        amplitude:
            The amount of each compound (the height of the peaks).
        asymmetry:
            Optional asymmetry to the peak shapes, e.g., fronting or tailing.
        noise_level:
            The magnitude of the random normal noise (standard deviation).
        dtype:
            The dtype of the output (chromatogram).
    '''

    def __init__(
        self,
        time: Time,
        loc: np.ndarray,
        scale: np.ndarray = 0.025,
        amplitude: np.ndarray = 1.0,
        asymmetry: np.ndarray = 0.0,
        noise_level: float = 0.005,
        dtype: np.dtype = np.float32,
    ):
        
        self.time = time 

        self.loc = loc.astype(np.float32)

        if isinstance(scale, float):
            scale = np.array([scale] * len(loc))
        if isinstance(amplitude, float):
            amplitude = np.array([amplitude] * len(loc))
        if isinstance(asymmetry, float):
            asymmetry = np.array([asymmetry] * len(loc))

        self.scale = scale.astype(np.float32)
        self.amplitude = amplitude.astype(np.float32) 
        self.asymmetry = asymmetry.astype(np.float32)

        eluted = np.where(np.isfinite(self.loc))[0]
        
        self._eluted = len(eluted)
        self._uneluted = len(self.loc) - len(eluted)

        self.loc = self.loc[eluted]
        self.scale = self.scale[eluted]
        self.amplitude = self.amplitude[eluted]
        self.asymmetry = self.asymmetry[eluted]
        self.noise_level = noise_level
        self.dtype = dtype
            
    def observation(self, seed: int = None) -> np.ndarray:

        '''The observation.

        This is what the agent observes when interacting with the environment.

        Args:
            seed: 
                The random seed for reproducability. Current unused.

        Returns:
            A chromatogram.
        '''

        rng = np.random.RandomState(seed)
    
        x = self.time.array

        scales = [self.scale, self.asymmetry]

        scales = np.sum([
            scale * (x - self.loc) ** n 
            for (n, scale) in enumerate(scales)], axis=0)

        peaks = self.amplitude * np.exp(-0.5*((x-self.loc)/scales) ** 2)

        chromatogram = np.sum(peaks, axis=1)

        noise = np.random.randn(len(chromatogram)) * self.noise_level 

        chromatogram += noise
        
        return np.expand_dims(chromatogram, axis=1).astype(self.dtype)

    @property
    def eluted(self):
        return self._eluted 
    
    @property 
    def uneluted(self):
        return self._uneluted
    

class ChromatographicResponse(NamedTuple):

    '''Temporarily implemented to return auxiliary data (resolutions).'''

    reward: np.ndarray 
    resolutions: np.ndarray


class ChromatographicResponseFunction:

    '''A chromatographic response function (CRF).

    Args:
        reward_coef:
            The weight applied to the reward term of the reward function.
        penalty_coef:
            The weight applied to the penalty term of the reward function.
            Currently not used.
        resolution_target:
            The target resolution (for each peak pair). Default to 1.5, which
            is considered enough for two adjacent peaks to be perfectly 
            separated.
        time_target:
            The time window for which we desire the compounds to elute.
            Default to 20.0, which means we want the compounds (peak locations)
            to be equal or less than 20 (minutes).
        dtype:
            The dtype of the output (reward).
        
    '''

    def __init__(
        self, 
        reward_coef: float = 1.00, 
        penalty_coef: float = 0.00, 
        resolution_target: float = 1.5, 
        time_target: float = 20.0,
        dtype: np.dtype = np.float32,
    ):
        self.reward_coef = reward_coef
        self.penalty_coef = penalty_coef
        self.resolution_target = resolution_target
        self.time_target = time_target
        self.dtype = dtype

    def __call__(self, separation: Separation) -> ChromatographicResponse:

        '''Given the current separation, computes reward.

        The reward is a type of chromatographic response function (CRF) which 
        indicates how good the separation is.
        
        Args:
            separation:
                The current separation.

        Returns:
            The reward
        '''

        total_components = (separation.eluted + separation.uneluted) 

        eluted_before_time_target = np.where(
            separation.loc <= self.time_target, 1.0, 0.0).sum()
        
        weight = eluted_before_time_target / total_components
        
        weight = weight * 2.0 - 1.0

        if separation.eluted < 2:
            return ChromatographicResponse(
               reward=np.array(1.0 * weight, self.dtype),
               resolutions=0)
        
        resolution = self._compute_resolution(separation.loc, separation.scale)
        resolution_scaled = self._scale_resolution(resolution)
        resolution_avg = self._average_resolution(resolution_scaled)
        
        reward = resolution_avg

        penalty = 0.0

        reward = (
            self.reward_coef * reward - self.penalty_coef * penalty
        )

        return ChromatographicResponse(
            reward=np.array(reward * weight, self.dtype),
            resolutions=resolution
        )
        
    def _compute_resolution(self, loc, scale):
        index = np.argsort(loc)
        loc_sorted = loc[index]
        scale_sorted = scale[index]
        pair_distance = loc_sorted[1:] - loc_sorted[:-1]
        pair_combined_width = scale_sorted[1:] * 2 + scale_sorted[:-1] * 2
        resolution = pair_distance / pair_combined_width
        return np.clip(resolution, 0, self.resolution_target)

    def _scale_resolution(self, resolution):
        return resolution / self.resolution_target * 2.0 - 1.0
    
    def _average_resolution(self, resolution):
        # return np.average(resolution, weights=np.exp(-resolution))
        return np.mean(resolution)
    