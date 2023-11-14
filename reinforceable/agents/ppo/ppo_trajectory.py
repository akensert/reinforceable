from typing import NamedTuple

from reinforceable.types import Tensor 
from reinforceable.types import NestedTensor
from reinforceable.types import NestedDistribution


class PPOTrajectory(NamedTuple):

    '''PPO trajectory.
    
    The processed trajectory for the PPO implementation. Namely, the 
    PPOTrajectory is the result of `finalize_trajectory`, and will be 
    passed as training data to `train`.

    '''
    
    state: NestedTensor
    step_type: Tensor
    action: NestedTensor
    action_mask: NestedTensor
    action_log_prob: Tensor
    value_pred: Tensor
    value_true: Tensor
    advantage: Tensor
    distrib: NestedDistribution
    trajectory_mask: Tensor
