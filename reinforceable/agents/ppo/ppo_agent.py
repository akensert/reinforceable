import tensorflow as tf

import keras

from reinforceable.agents.agent import Agent

from reinforceable.types import Tensor
from reinforceable.types import NestedTensor
from reinforceable.types import NestedDistribution
from reinforceable.types import TrainInfo

from reinforceable.timestep import Timestep
from reinforceable.trajectory import Trajectory

from reinforceable.agents.ppo.ppo_trajectory import PPOTrajectory
from reinforceable.agents.ppo.ppo_normalizer import StreamingNormalizer 
from reinforceable.agents.ppo import ppo_ops
from reinforceable.agents.ppo import ppo_losses
from reinforceable.agents.ppo import ppo_callbacks


from reinforceable.utils.nested_ops import expand_dim
from reinforceable.utils.nested_ops import squeeze_dim


class RecurrentPPOAgent(Agent):

    '''A recurrent proximal policy optimization (PPO) agent.
    
    The agent uses recurrent neural networks to pass hidden states
    between time steps, allowing the agent to make decisions based on 
    historical time steps.
    
    Args:
        encoder_network:
            ...
        policy_network:
            ...
        value_network:
            ...
        optimizer:
            ...
        discount_factor:
            ...
        lambda_factor:
            ...
        use_gae:
            ...
        use_td_lambda_return:
            ...
        policy_loss_coef:
            ...
        value_loss_coef:
            ...
        entropy_loss_coef:
            ...
        reg_loss_coef:
            ...
        kl_cutoff_factor:
            ...
        kl_cutoff_coef:
            ...
        kl_beta_initial:
            ...
        kl_target:
            ...
        kl_tolerance:
            ...
        advantage_normalization:
            ...
        reward_normalization:
            ...
        state_normalization:
            ...
        importance_ratio_clip:
            ...
        value_clip:
            ...
        reward_clip:
            ...
        state_clip:
            ...
        log_prob_clip:
            ...
        gradient_clip:
            ...
        train_callbacks:
            ...
        train_eagerly:
            ...
        summary_writer:
            ...
        name:
            ...
    '''

    def __init__(
        self,
        encoder_network: tf.keras.Model,
        policy_network: tf.keras.Model,
        value_network: tf.keras.Model,
        *,
        optimizer: keras.optimizers.Optimizer = None,
        discount_factor: float = 0.99,
        lambda_factor: float = 0.95,
        use_gae: bool = True,
        use_td_lambda_return: bool = False,
        policy_loss_coef: float = 1.0,
        value_loss_coef: float = 0.5,
        entropy_loss_coef: float = 0.01,
        reg_loss_coef: float|tuple[float, float, float] = 0.0,
        kl_cutoff_factor: float = 2.0,
        kl_cutoff_coef: float = 1000.0,
        kl_beta_initial: float = 1.0,
        kl_target: float = 0.01,
        kl_tolerance: float = 0.3,
        advantage_normalization: bool = True,
        reward_normalization: bool = True,
        state_normalization: bool = True,
        importance_ratio_clip: float = 0.2,
        value_clip: float = 0.2,
        reward_clip: float = 10.0,
        state_clip: float = 10.0,
        log_prob_clip: float = None,
        gradient_clip: float = None,
        train_callbacks: list[keras.callbacks.Callback] = None,
        train_eagerly: bool = False,
        summary_writer: str|tf.summary.SummaryWriter = None,
        name: str = 'RecurrentPPOAgent',
    ) -> None:

        train_callbacks = [
            ppo_callbacks.HiddenStateCallback(), 
            ppo_callbacks.AdaptiveKLBetaCallback()
        ] + [] if train_callbacks is None else train_callbacks

        super().__init__(
            train_callbacks=train_callbacks, 
            train_eagerly=train_eagerly, 
            summary_writer=summary_writer,
            name=name
        )

        self.encoder_network = encoder_network
        self.policy_network = policy_network
        self.value_network = value_network

        self.discount_factor = discount_factor
        self.lambda_factor = lambda_factor
        self.use_td_lambda_return = use_td_lambda_return
        self.use_gae = True if use_td_lambda_return else use_gae
        self._advantage_normalization = advantage_normalization

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(2.5e-4)

        self.optimizer = tf.keras.optimizers.get(optimizer)

        # Loss related 
        self.policy_loss_coef = policy_loss_coef
        self.value_loss_coef = value_loss_coef
        self.reg_loss_coef = reg_loss_coef or 0.0
        self.entropy_loss_coef = entropy_loss_coef
        self.kl_cutoff_factor = kl_cutoff_factor
        self.kl_cutoff_coef = kl_cutoff_coef
        self.kl_target = kl_target
        self.kl_tolerance = kl_tolerance
        self.value_clip = value_clip or 0.0
        self.importance_ratio_clip = importance_ratio_clip or 0.0
        self.log_prob_clip = log_prob_clip or 0.0

        self.gradient_clip = gradient_clip or 0.0

        with self.name_scope:
            
            if kl_beta_initial is None or kl_beta_initial == 0:
                self.adaptive_kl_beta = None 
            else:
                self.adaptive_kl_beta = tf.Variable(
                    initial_value=kl_beta_initial, 
                    dtype=tf.float32, 
                    trainable=False)

            self._reward_normalizer = (
                None if not reward_normalization else 
                StreamingNormalizer(clip_value=reward_clip, center_mean=False)
            )
            self._state_normalizer = (
                None if not state_normalization else 
                StreamingNormalizer(clip_value=state_clip, center_mean=True)
            )

        self.finalize_trajectory = (
            self.finalize_trajectory if self.train_eagerly else 
            tf.function(self.finalize_trajectory))
        
    def __call__(
        self, 
        timestep: Timestep, 
        *,
        deterministic: bool = False,
        training: bool = False,
    ) -> tuple[NestedTensor, dict[str, Tensor]]:
        
        '''Computes an action, given the current timestep.

        In addition to the action being returned, additional information is
        also returned, such as the encoded state, state value, and action 
        log probablities.

        Args:
            timestep:
                The current timestep supplied by the environment.
            deterministic:
                Whether the action should be deterministic or not. If
                determinsitic, `mode()` is invoked, otherwise `sample()`.
                Default to False.
            training:
                Whether to flag training or not. Default to False.
            
        Returns:
            The actions as well as auxiliary information.
        '''

        if self._state_normalizer is None:
            state = timestep.state 
        else:
            state = self._state_normalizer.normalize(timestep.state)

        state = (state, timestep.step_type != 0)

        encoded_state = self.encoder_network(state, training=training)

        value = self.value_network(encoded_state, training=training)

        if 'action_mask' in timestep.info:
            encoded_state = (encoded_state, timestep.info['action_mask'])

        distrib = self.policy_network(encoded_state, training=training)
        action = ppo_ops.compute_action(distrib, deterministic=deterministic)
        log_prob = ppo_ops.compute_log_prob(distrib, action, batch_ndims=2)

        aux_info = {
            'encoded_state': encoded_state,
            'value': value,
            'action_log_prob': log_prob,
        }
        
        return action, aux_info
    
    def train_step(
        self, 
        data: PPOTrajectory|tuple[PPOTrajectory, Tensor],
        sample_weight: Tensor = None,
    ) -> TrainInfo:

        '''A training step.
        
        Based on trajectory data, a number of losses are calculated, 
        including the policy loss, value loss and entropy loss. Subsequently, 
        with respect to these losses, the agent updates its weights; i.e., its 
        encoder network, policy network and value network.

        Args:
            data:
                A chunk (or segment) of a trajectory.

        Returns:
            Training info.
        '''

        training = True

        tf.summary.scalar('learning_rate', self.optimizer.learning_rate)

        with tf.GradientTape() as tape:
            
            # State was normalized in `preprocess`
            state = (data.state, data.step_type != 0)
            encoded_state = self.encoder_network(state, training=training)

            value = self.value_network(encoded_state, training=training)

            if data.action_mask is not None:
                encoded_state = (encoded_state, data.action_mask)

            distrib = self.policy_network(encoded_state, training=training)

            if sample_weight is not None:
                # Perform trajectory masking outside to avoid doing it four 
                # times over later on.
                sample_weight = tf.boolean_mask(
                    sample_weight, data.trajectory_mask)
            
            policy_loss = ppo_losses.policy_loss(
                distrib=distrib, 
                action=data.action,
                action_log_prob=data.action_log_prob,
                advantage=data.advantage,
                importance_ratio_clip=self.importance_ratio_clip,
                log_prob_clip=self.log_prob_clip,
                loss_coef=self.policy_loss_coef,
                sample_mask=data.trajectory_mask,
                sample_weight=sample_weight,
            )
            value_loss = ppo_losses.value_loss(
                value=value,
                value_true=data.value_true,
                value_pred=data.value_pred,
                value_clip=self.value_clip,
                loss_coef=self.value_loss_coef,
                sample_mask=data.trajectory_mask,
                sample_weight=sample_weight,
            )
            entropy_loss = ppo_losses.entropy_loss(
                distrib=distrib,
                loss_coef=self.entropy_loss_coef,
                sample_mask=data.trajectory_mask,
                sample_weight=sample_weight,
            )
            kl_mean, adaptive_kl_loss, cutoff_kl_loss = ppo_losses.kl_loss(
                distrib=distrib,
                old_distrib=data.distrib,
                kl_target=self.kl_target,
                kl_cutoff_factor=self.kl_cutoff_factor,
                kl_cutoff_coef=self.kl_cutoff_coef,
                adaptive_kl_beta=self.adaptive_kl_beta,
                sample_mask=data.trajectory_mask,
                sample_weight=sample_weight,
            )
            reg_loss = ppo_losses.reg_loss(
                self.encoder_network.trainable_variables,
                self.policy_network.trainable_variables, 
                self.value_network.trainable_variables,
                coefs=self.reg_loss_coef
            )
            loss = (
                policy_loss + value_loss + entropy_loss + 
                adaptive_kl_loss + cutoff_kl_loss + reg_loss
            )
                                    
        variables = (
            self.encoder_network.trainable_variables + 
            self.policy_network.trainable_variables + 
            self.value_network.trainable_variables
        )

        gradients = tape.gradient(loss, variables)

        if self.gradient_clip > 0.0:
            gradients, _ = tf.clip_by_global_norm(
                gradients, self.gradient_clip)

        # tf.summary.histogram for gradients ?

        self.optimizer.apply_gradients(zip(gradients, variables))

        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'adaptive_kl_loss': adaptive_kl_loss,
            'cutoff_kl_loss': cutoff_kl_loss,
            'regularization_loss': reg_loss,
            'kl_mean': kl_mean,
        }
    
    def finalize_trajectory(
        self, 
        trajectory: Trajectory,
        final_timestep: Timestep,
    ) -> PPOTrajectory:
        
        '''Preprocesses trajectory data.
        
        Converts trajectory data to PPO relevant trajectory data.

        In addition to typical trajectory data such as states, step types,
        actions, action masks and rewards, action log probablities, target 
        values, predicted values, advantage estimations and distributions are 
        computed for the PPO agent to learn from.

        Args:
            trajectory (Trajectory):
                A trajectory of `Timestep`s obtained during the 
                agent-environment interaction. Importantly, each data 
                component of the `Trajectory` is time major. Namely, states,
                step types, actions, etc., each have the shape (T, B, ...).
                Each row (t in {0, ..., T}) corresponds to a `Timestep`.

        Returns:
            PPOTrajectory: training data, for both the policy and value 
            optimization, as well as entropy and kl divergence regularization.  
        '''

        training = False 

        state = trajectory.state

        value = trajectory.info['value']
        
        final_timestep = expand_dim(final_timestep, 0)

        final_value = self._compute_final_value(
            final_timestep, training=training)

        next_value = tf.concat([value[1:], final_value], axis=0)
        
        # Reward shifted left as we want the reward of the next timestep.
        reward = tf.concat([
            trajectory.reward[1:], final_timestep.reward], axis=0)
        
        # Discount shifted left as we want to discount the next value.
        discount = tf.concat([
            trajectory.discount(self.discount_factor)[1:], 
            final_timestep.discount(self.discount_factor)], axis=0)

        if self._reward_normalizer is not None:
            reward = self._reward_normalizer.normalize(reward)

        advantages, returns = self._compute_advantages_and_returns(
            reward, discount, value, next_value)

        if self._state_normalizer is not None:
            state = self._state_normalizer.normalize(state)

        # Trajectory mask to ignore boundaries in episodes.
        # For instance, we do not care about the action and subsquent 
        # advantage in a terminal state.
        trajectory_mask = trajectory.step_type != 2

        if self._advantage_normalization:
            advantages = ppo_ops.normalize_advantage(
                advantages, trajectory_mask)

        # Update state normalizer
        if self._state_normalizer is not None:
            self._state_normalizer.update(trajectory.state, reduce_ndims=2)

        # Update reward normalizer
        if self._reward_normalizer is not None:
            # Mask out rewards of initial states as they are irrelevant.
            # Rewards are only relevant for states that resulted from actions.
            self._reward_normalizer.update(
                tf.boolean_mask(trajectory.reward, trajectory.step_type != 0),
                reduce_ndims=1)

        return PPOTrajectory(
            state=state,
            step_type=trajectory.step_type,
            action=trajectory.info['action'],
            action_mask=trajectory.info.get('action_mask', None),
            action_log_prob=trajectory.info['action_log_prob'],
            value_pred=value,
            value_true=returns,
            advantage=advantages,
            trajectory_mask=trajectory_mask,
            distrib=self._recover_distribution(trajectory, training=training)
        )

    def _compute_final_value(
        self,
        final_timestep: Timestep,
        training: bool = False,
    ) -> Tensor:
        state = final_timestep.state
        if self._state_normalizer is not None:
            state = self._state_normalizer.normalize(state)
        state = (state, final_timestep.step_type != 0)
        encoded_state = self.encoder_network(state, training=training)
        value = self.value_network(encoded_state, training=training)
        return value * final_timestep.discount(self.discount_factor)

    def _compute_advantages_and_returns(
        self, 
        reward: Tensor, 
        discount: Tensor, 
        value: Tensor, 
        next_value: Tensor
    ) -> tuple[Tensor, Tensor]:
        
        if self.use_gae:
            advantages = ppo_ops.compute_advantage(
                reward, discount, value, next_value, self.lambda_factor)
            if self.use_td_lambda_return:
                returns = (advantages + value)
        
        if not self.use_td_lambda_return:
            returns = ppo_ops.compute_return(reward, discount, next_value[-1])
            if not self.use_gae:
                advantages = (returns - value)

        return advantages, returns
    
    def _recover_distribution(
        self, 
        trajectory: Trajectory,
        training: bool = False,
    ) -> NestedDistribution:
        
        if 'action_mask' not in trajectory.info:
            distrib = self.policy_network(
                trajectory.info['encoded_state'], training=training) 
        else:
            distrib = self.policy_network(
                (trajectory.info['encoded_state'], 
                 trajectory.info['action_mask']), 
                training=training)

        # Extracts tensor_distribution from _TensorCoercible. Is this fine?
        distrib = tf.nest.map_structure(
            lambda x: x.tensor_distribution, distrib)

        return distrib

    def save(self, path, *args, **kwargs):
        kwargs['deterministic'] = kwargs.pop(
            'deterministic', tf.TensorSpec([], tf.bool, name='deterministic'))
        kwargs['training'] = kwargs.pop(
            'training', tf.TensorSpec([], tf.bool, name='training'))
        super().save(path, *args, **kwargs)