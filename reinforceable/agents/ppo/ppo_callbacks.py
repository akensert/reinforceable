import tensorflow as tf

from keras.callbacks import Callback 

from reinforceable.layers.stateful_rnn import StatefulRNN

# TODO: should methods such as on_batch_begin be replaced with on_batch_train_begin?
#       As of now, custom train method of agent invokes on_batch_begin etc.

class HiddenStateCallback(Callback):

    '''Assures that hidden states are reset and updated approprietly.'''

    def on_train_begin(self, logs=None):
        if not hasattr(self, 'states'):
            self.states = []
            for layer in self.model.encoder_network.layers:
                if isinstance(layer, StatefulRNN):
                    self.states.append(layer._state)

            self.initial_states = tf.nest.map_structure(
                lambda x: tf.Variable(
                    x.value(), dtype=x.dtype, trainable=x.trainable),
                self.states)

    def on_epoch_begin(self, epoch, logs=None):
        tf.nest.map_structure(
            lambda dst, src: dst.assign(src), self.states, self.initial_states)
    
    def on_train_end(self, logs=None):
        tf.nest.map_structure(
            lambda dst, src: dst.assign(src), self.initial_states, self.states)


class AdaptiveKLBetaCallback(Callback):
    
    '''Adapts KL Beta for the adaptive KL loss.'''
    
    def on_train_begin(self, logs=None):
        if not hasattr(self, 'update_kl_beta'):
            self.update_kl_beta = self.model.adaptive_kl_beta is not None

    def on_epoch_begin(self, epoch, logs=None):
        if self.update_kl_beta:
            self.divisor = 0.0
            self.kl_divergence = 0.0

    def on_batch_end(self, batch, logs=None):
        if self.update_kl_beta:
            self.kl_divergence += logs['kl_mean']
            self.divisor += 1.0

    def on_train_end(self, logs=None):
        if self.update_kl_beta:
            self._adapt_kl_beta()

    def _adapt_kl_beta(self):

        mean_kl = tf.constant(self.kl_divergence / self.divisor)

        mean_kl_below_bound = (
            mean_kl < self.model.kl_target * (1.0 - self.model.kl_tolerance)
        )
        mean_kl_above_bound = (
            mean_kl > self.model.kl_target * (1.0 + self.model.kl_tolerance)
        )
        adaptive_kl_update_factor = tf.case(
            [
                (
                    mean_kl_below_bound,
                    lambda: tf.constant(1.0 / 1.5, dtype=tf.float32)
                ),
                (
                    mean_kl_above_bound, 
                    lambda: tf.constant(1.5, dtype=tf.float32)
                ),
            ],
            default=lambda: tf.constant(1.0, dtype=tf.float32),
            exclusive=True
        )
        new_adaptive_kl_beta = tf.clip_by_value(
            self.model.adaptive_kl_beta * adaptive_kl_update_factor,
            clip_value_min=10e-16,
            clip_value_max=10e+16)
        
        self.model.adaptive_kl_beta.assign(new_adaptive_kl_beta)

