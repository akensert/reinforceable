import tensorflow as tf

from typing import Sequence

from reinforceable.types import Tensor 
from reinforceable.types import NestedTensor 
from reinforceable.types import NestedDistribution

from reinforceable.agents.ppo import ppo_ops


# TODO: Combine sample mask (trajectory mask) with sample weights?

def policy_loss(
    distrib: NestedDistribution,
    action: NestedTensor,
    action_log_prob: Tensor,
    advantage: Tensor,
    importance_ratio_clip: float,
    log_prob_clip: float,
    loss_coef: float,
    sample_mask: Tensor,
    sample_weight: Tensor,
) -> float:

    '''The policy loss.

    Args:
        distrib:
            Newly computed distribution(s) by the agent (policy network).
        action:
            The actions from the buffer. Corresponds to `distrib` input.
        action_log_prob:
            The log probablity of the actions from the buffer.
        advantage:
            The advantage from the buffer.
        importance_ratio_clip:
            A float value specifying the clip range for clipping the importance ratio.
        log_prob_clip:
            A float vlaue specifying the clip range for the log prob
        loss_coef:
            A scalar float to weight the policy loss.
        sample_mask:
            Trajectory mask.
        sample_weight:
            Sample weights.

    Returns:
        The policy loss.
    '''

    log_prob = ppo_ops.compute_log_prob(distrib, action, batch_ndims=2)

    if log_prob_clip > 0.0:
        log_prob_clip = (-log_prob_clip, log_prob_clip)
        log_prob = tf.clip_by_value(log_prob, *log_prob_clip)
        action_log_prob = tf.clip_by_value(action_log_prob, *log_prob_clip)
    
    ratio = tf.exp(log_prob - action_log_prob)

    ratio = tf.boolean_mask(ratio, sample_mask)
    advantage = tf.boolean_mask(advantage, sample_mask)

    advantage_ratioed = ratio * advantage

    if importance_ratio_clip > 0.0:
        ratio_clip = (1 - importance_ratio_clip, 1 + importance_ratio_clip)
        ratio_clipped = tf.clip_by_value(ratio, *ratio_clip)
        advantage_ratioed_clipped = advantage * ratio_clipped
        advantage_ratioed = tf.minimum(
            advantage_ratioed, advantage_ratioed_clipped)
    
    if sample_weight is not None:
        advantage_ratioed = advantage_ratioed * sample_weight

    policy_loss = - tf.reduce_mean(advantage_ratioed) * loss_coef

    tf.summary.histogram('values/log_probs', log_prob)
    tf.summary.histogram('values/advantages', advantage)
    tf.summary.scalar('losses/policy_loss', policy_loss)

    return policy_loss

def value_loss(
    value: Tensor, 
    value_true: Tensor,
    value_pred: Tensor,
    value_clip: float,
    loss_coef: float,
    sample_mask: Tensor,
    sample_weight: Tensor,
) -> float:
    
    '''The value loss.

    Args:
        value:
            Newly predicted values by the agent (value network).
        value_true:
            The target value (return) computed from the buffer.
        value_pred:
            The predicted value computed from the buffer.
        value_clip:
            A float value specifying the clip range. Namely, (-value_clip, value_clip).
        loss_coef:
            A scalar float to weight the policy loss.
        sample_mask:
            Trajectory mask. 
        sample_weight:
            Sample weights.
    
    Returns:
        The value loss.
    '''
    
    value = tf.boolean_mask(value, sample_mask)
    value_pred = tf.boolean_mask(value_pred, sample_mask)
    value_true = tf.boolean_mask(value_true, sample_mask)

    value_error = tf.math.squared_difference(value, value_true)

    if value_clip > 0.0:
        value_clip =  (-value_clip, value_clip)
        value_clipped = (
            value_pred + tf.clip_by_value(value - value_pred, *value_clip)
        )
        value_clipped_error = tf.math.squared_difference(
            value_clipped, value_true
        )
        value_error = tf.maximum(value_error, value_clipped_error)

    if sample_weight is not None:
        value_error = value_error * sample_weight

    value_loss = tf.reduce_mean(value_error) * loss_coef

    tf.summary.histogram('values/values', value)
    tf.summary.histogram('values/returns', value_true)
    tf.summary.scalar('losses/value_loss', value_loss)

    return value_loss 

def entropy_loss(
    distrib: NestedDistribution,
    loss_coef: float,
    sample_mask: Tensor,
    sample_weight: Tensor,
) -> float:
    
    '''The entropy loss.

    Args:
        distrib:
            Newly computed distribution(s) by the agent (policy network).
        loss_coef:
            A scalar float to weight the policy loss.
        sample_mask:
            Trajectory mask. 
        sample_weight:
            Sample weights.

    Returns:
        The entropy loss.
    '''
    if loss_coef <= 0.0:
        return tf.constant(0.0)
    
    entropy = ppo_ops.compute_entropy(distrib, batch_ndims=2)

    entropy = tf.expand_dims(entropy, axis=-1)
    entropy = tf.boolean_mask(entropy, sample_mask)
    
    if sample_weight is not None:
        entropy = entropy * sample_weight

    entropy_loss = - tf.reduce_mean(entropy) * loss_coef

    tf.summary.histogram('values/entropies', entropy)
    tf.summary.scalar('losses/entropy_loss', entropy_loss)

    return entropy_loss

def kl_loss(
    distrib: NestedDistribution,
    old_distrib: NestedDistribution,
    kl_target: float, 
    kl_cutoff_factor: float,
    kl_cutoff_coef: float,
    adaptive_kl_beta: tf.Variable,
    sample_mask: Tensor,
    sample_weight: Tensor,
) -> tuple[float, float, float]:
    
    '''The entropy loss.

    Args:
        distrib:
            The new (latest) distribution(s) computed by the policy network.
        old_distrib: 
            The old (initial) distribution(s), computed earlier by the policy
            network.
        kl_target:
            The (scalar) target for the kl divergence.
        kl_cutoff_factor:
            The weight multiplied with the kl target.
        kl_cutoff_coef:
            The weight multiplied with the kl cutoff loss.
        adaptive_kl_beta:
            The weight multiplied with the kl loss.
        sample_mask:
            Trajectory mask. 
        sample_weight:
            Sample weights.
    Returns:
        The mean kl divergence as well as kl losses.
    '''

    if kl_cutoff_factor > 0.0 or adaptive_kl_beta is not None:
        kl_divergence = ppo_ops.compute_kl_divergence(
            old_distrib, distrib, batch_ndims=2)
        kl_divergence = tf.expand_dims(kl_divergence, axis=-1)
        kl_divergence = tf.boolean_mask(kl_divergence, sample_mask)
        if sample_weight is not None:
            kl_divergence = kl_divergence * sample_weight
        mean_kl_divergence = tf.reduce_mean(kl_divergence)
    else:
        mean_kl_divergence = tf.constant(0.0)

    if adaptive_kl_beta is not None:
        adaptive_kl_loss = (mean_kl_divergence * adaptive_kl_beta)
    else:
        adaptive_kl_loss = tf.constant(0.0)

    if kl_cutoff_factor > 0.0:
        kl_cutoff = (kl_cutoff_factor * kl_target)
        cutoff_kl_loss = tf.maximum(mean_kl_divergence - kl_cutoff, 0.0)
        cutoff_kl_loss = tf.square(cutoff_kl_loss) * kl_cutoff_coef
    else:
        cutoff_kl_loss = tf.constant(0.0)

    tf.summary.histogram('values/kl_divergences', kl_divergence)
    tf.summary.scalar('losses/mean_kl_divergence', mean_kl_divergence)
    tf.summary.scalar('losses/cutoff_kl_loss', cutoff_kl_loss)
    tf.summary.scalar('losses/adaptive_kl_loss', adaptive_kl_loss)
    tf.summary.scalar('losses/kl_beta', adaptive_kl_beta)

    return mean_kl_divergence, adaptive_kl_loss, cutoff_kl_loss

def reg_loss(
    encoder_variables: list[tf.Variable],
    policy_variables: list[tf.Variable],
    value_variables: list[tf.Variable],
    coefs: tuple[float, float, float], # encoder, policy, value
) -> float:
    
    '''L2 regularization loss (also known as weight decay).
    
    Args:
        encoder_variables:
            The trainable variables (weights) of the encoder network.
        policy_variables:
            The trainable variables (weights) of the policy network.
        value_variables:
            The trainable variables (weights) of the value network.
        coefs:
            Coefficients applied to encoder, policy and value penalties.

    Returns:
        The regularization loss.
    '''
    
    if not isinstance(coefs, Sequence):
        coefs = [coefs] * 3

    if not any(coefs):
        return tf.constant(0.0)

    encoder_vars = [v for v in encoder_variables if 'kernel' in v.name]
    policy_vars = [v for v in policy_variables if 'kernel' in v.name]
    value_vars = [v for v in value_variables if 'kernel' in v.name]

    return tf.add_n(
          [tf.reduce_sum(tf.square(v)) * coefs[0] for v in encoder_vars] 
        + [tf.reduce_sum(tf.square(v)) * coefs[1] for v in policy_vars] 
        + [tf.reduce_sum(tf.square(v)) * coefs[2] for v in value_vars]
    )

