import tensorflow as tf

from reinforceable.types import Tensor 
from reinforceable.types import NestedTensor
from reinforceable.types import Distribution
from reinforceable.types import NestedDistribution


def compute_action(
    distribs: NestedDistribution,
    deterministic: bool = False,
    expand_composites: bool = False,
) -> tf.Tensor|tuple[tf.Tensor, ...]:
    
    '''Samples action(s) from distribution object(s).
    
    Args: 
        distribs: 
            Distribution objects.
        deterministic:
            Whether to sample determinstically: `sample()` vs `mode()`.
        expand_composites:
            Whether to expand composites with the tf.nest API.

    Returns:
        A set of actions corresponding to distribs. Has the same structure
        as distribs: `tf.nest.assert_same_structure(actions, distribs)`
    '''

    return tf.nest.map_structure(
        lambda dist: dist.sample() if not deterministic else dist.mode(),
        distribs, expand_composites=expand_composites)

def compute_log_prob(
    distribs: NestedDistribution,
    actions: NestedTensor,
    expand_composites: bool = False,
    batch_ndims: int = 2,
) -> tf.Tensor:
    
    '''Computes log probability of action(s).
    
    A single log probability value is obtained for each sample in the batch; 
    meaning that event dimension as well as the different distributions
    (corresponding to a single sample) are summed. 

    Args: 
        distribs: 
            Distribution objects.
        actions:
            Actions.
        expand_composites:
            Whether to expand composites with the tf.nest API.
        batch_ndims:
            Number of batch dimensions in the data.

    Returns:
        Log probability corresponding to each sample.
    '''

    def log_prob_fn(distrib: Distribution, action: Tensor) -> Tensor:
        log_prob = distrib.log_prob(action)
        rank = log_prob.shape.rank
        reduce_dims = list(range(batch_ndims, rank))
        log_prob = tf.reduce_sum(log_prob, axis=reduce_dims, keepdims=True)
        return tf.expand_dims(log_prob, -1)

    tf.nest.assert_same_structure(
        distribs, actions, expand_composites=expand_composites)
    distribs = tf.nest.flatten(
        distribs, expand_composites=expand_composites)
    actions = tf.nest.flatten(
        actions, expand_composites=expand_composites)
    log_probs = tf.nest.map_structure(log_prob_fn, distribs, actions)
    return tf.add_n(log_probs)

def compute_entropy(
    distribs: NestedDistribution,
    expand_composites: bool = False,
    batch_ndims: int = 2,
) -> tf.Tensor:
    
    '''Computes entropy of distribution(s).

    A single entropy value is obtained for each sample in the batch; meaning 
    that the different distributions (corresponding to a single sample) are 
    summed. 
    
    Args: 
        distribs: 
            Distribution objects.
        expand_composites:
            Whether to expand composites with the tf.nest API.
        batch_ndims:
            Number of batch dimensions in the data.

    Returns:
        Entropy corresponding to each sample.

    '''

    def entropy_fn(distrib: Distribution) -> Tensor:
        entropy = distrib.entropy()
        rank = entropy.shape.rank
        reduce_dims = list(range(batch_ndims, rank))
        return tf.reduce_sum(entropy, axis=reduce_dims, keepdims=True)

    distribs = tf.nest.flatten(distribs, expand_composites=expand_composites)
    entropies = tf.nest.map_structure(entropy_fn, distribs)
    return tf.add_n(entropies) # batch dim not reduced

def compute_kl_divergence(
    from_distribs: NestedDistribution, 
    to_distribs: NestedDistribution,
    expand_composites: bool = False,
    batch_ndims: int = 2,
) -> tf.Tensor:
    
    '''Computes kl divergence(s) between pair(s) of distributions.
    
    A single kl divergence value is obtained for each sample in the batch; 
    meaning that the kl divergence of the different pairs of distributions 
    (corresponding to a single sample) are summed. 
    
    Args: 
        from_distribs: 
            Distribution objects.
        to_distribs: 
            Distribution objects.
        expand_composites:
            Whether to expand composites with the tf.nest API.
        batch_ndims:
            Number of batch dimensions in the data.

    Returns:
        KL divergence corresponding to each sample. 
    '''

    def kl_divergence_fn(
        from_distrib: Distribution, 
        to_distrib: Distribution
    ) -> Tensor:
        kl_divergence = from_distrib.kl_divergence(to_distrib)
        rank = kl_divergence.shape.rank
        reduce_dims = list(range(batch_ndims, rank))
        return tf.reduce_sum(kl_divergence, axis=reduce_dims, keepdims=True)

    tf.nest.assert_same_structure(
        from_distribs, to_distribs, expand_composites=expand_composites)
    from_distribs = tf.nest.flatten(
        from_distribs, expand_composites=expand_composites)
    to_distribs = tf.nest.flatten(
        to_distribs, expand_composites=expand_composites)
    kl_divergences = tf.nest.map_structure(
        kl_divergence_fn, from_distribs, to_distribs)
    return tf.add_n(kl_divergences) # batch dim not reduced
    
def compute_return(
    reward: Tensor, 
    discount: Tensor, 
    last_value: Tensor,
) -> Tensor:
    
    '''Computes returns based on empirical data.
    
    Returns will serve as the target values when optimizing the value network.

    Args:
        reward:
            Rewards obtained from the agent-environment interaction.
        discount:
            Discounts, zero at terminals.
        last_value:
            The last value computed via the value network. Will serve as the
            initial value when computing the return (from end to beginning).

    Returns:
        Returns.
    '''

    return weighted_cumsum(
        values=reward,
        weights=discount,
        initializer=last_value,
        reverse=True
    )

def compute_advantage(
    reward: Tensor, 
    discount: Tensor, 
    value: Tensor, 
    next_value: Tensor, 
    variance_reduction: float
) -> Tensor:
    
    '''Computes advantages based on rewards and value predictions.
    
    Advantages are used to optimize the policy network; they determine whether 
    probabilities of actions should be increased or decreased.
    
    Args:
        reward:
            Rewards obtained from the agent-environment interaction.
        discount:
            Discounts, zero at terminals.
        value:
            The state values, predicted by the value network.
        next_value
            The next state values, predicted by the value network.
        variance_reduction:
            The lambda factor for generalized advantage estimation (GAE).
    
    Returns:
        Advantages.
    '''
    
    delta = reward + discount * next_value - value
    return weighted_cumsum(
        values=delta, 
        weights=(discount * variance_reduction), 
        initializer=None, 
        reverse=True
    )
    
def weighted_cumsum(
    values: Tensor, 
    weights: Tensor, 
    initializer: Tensor = None,
    reverse: bool = False,
    stop_gradient: bool = False,
) -> Tensor:
    
    '''Computes weighted cumulative sum over the first dimension.

    Similar to `tf.cumsum`, but with weights.
    
    Args:
        values:
            Any array of values that should be accumulated.
        weights:
            The weights to be applied to the values at each accumulation step.
            Should be broadcastable with values.
        initializer:
            The initial value.
        reverse:
            Whether to go from end to start, or start to end.
        stop_gradient:
            Whether to stop gradients from flowing.

    Returns:
        An array of values of the same shape as values, but with each value
        being the cumulative sum of preceeding values (or succeeding values 
        considering reverse traversal). 
    '''

    def fn(
        accumulated_value: tf.Tensor, 
        inputs: tuple[tf.Tensor, float]
    ) -> tf.Tensor:
        value, weight = inputs
        return value + weight * accumulated_value

    if initializer is None:
        initializer = tf.zeros_like(values[-1], dtype=values.dtype)

    output = tf.scan(
        fn=fn,
        elems=(values, weights),
        initializer=initializer,
        reverse=reverse)

    if stop_gradient:
        return tf.stop_gradient(output)

    return output

def normalize_advantage(
    advantage: Tensor, 
    mask: Tensor,
) -> Tensor:
    
    '''Normalizes advantages via centering and standard scaling.
    
    Args:
        advantage:
            The unnormalized advantages.
        mask:
            The trajectory mask.

    Returns:
        Normalized advantages.
    '''

    eps = tf.keras.backend.epsilon()
    adv_masked = tf.boolean_mask(advantage, mask)
    adv_mean = tf.math.reduce_mean(adv_masked, keepdims=True)
    adv_std = tf.math.reduce_std(adv_masked, keepdims=True)
    return (advantage - adv_mean) / tf.maximum(adv_std, eps)