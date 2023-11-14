import tensorflow as tf
import tensorflow_probability as tfp


class BoundedNormal(tfp.distributions.TransformedDistribution):

    _NUM_SAMPLES_FOR_ENTROPY_CALC = 100

    def __init__(
        self, 
        distribution: tfp.distributions.Distribution, 
        bounds: tf.Tensor, 
        validate_args: bool = False,
        parameters = None,
        name = 'BoundedNormal'
    ) -> 'BoundedNormal':
        parameters = dict(locals())
        action_means = (bounds[..., 1] + bounds[..., 0]) / 2.0
        action_magnitudes = (bounds[..., 1] - bounds[..., 0]) / 2.0
        bijector_chain = tfp.bijectors.Chain([
            tfp.bijectors.Shift(action_means)(
                tfp.bijectors.Scale(action_magnitudes)),
            _TanhBijector()
        ])      
        self._action_magnitudes = action_magnitudes
        self._action_means = action_means

        super().__init__(
            distribution=distribution, 
            bijector=bijector_chain, 
            validate_args=validate_args,
            name=name)

        self._parameters = parameters

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            distribution=tfp.util.ParameterProperties(),
            bijector=tfp.util.ParameterProperties(
                event_ndims=lambda td: tf.nest.map_structure( 
                    tf.rank, td.distribution.event_shape),
                event_ndims_tensor=lambda td: tf.nest.map_structure(
                    tf.size, td.distribution.event_shape_tensor())))
    
    def _entropy(self) -> tf.Tensor:
        sample = self.sample(self._NUM_SAMPLES_FOR_ENTROPY_CALC)
        logprob = self.log_prob(sample)
        return -tf.reduce_mean(logprob, 0)

    def _stddev(self, name: str = 'stddev') -> tf.Tensor:
        stddev = (
            self._action_magnitudes * tf.tanh(self._distribution.stddev())
        )
        return stddev

    def _mode(self, name: str = 'mode') -> tf.Tensor:
        mean = (
            self._action_magnitudes * tf.tanh(self._distribution.mode()) 
            + self._action_means
        )
        return mean

    def _mean(self, name: str = 'mean', **kwargs) -> tf.Tensor:
        return self.mode(name)


class _TanhBijector(tfp.bijectors.Bijector):

    def __init__(
        self, 
        validate_args: bool = False, 
        name: str = 'tanh_bijector'
    ) -> None:
        parameters = dict(locals())
        super().__init__(
            forward_min_event_ndims=0,
            validate_args=validate_args,
            parameters=parameters,
            name=name)

    def _forward(self, x: tf.Tensor) -> tf.Tensor:
        return tf.nn.tanh(x)

    def _inverse(self, y: tf.Tensor) -> tf.Tensor:
        clipped = tf.where(
            tf.less_equal(tf.abs(y), 1.),
            tf.clip_by_value(y, -0.99999997, 0.99999997), 
            y)
        return tf.atanh(clipped)

    def _forward_log_det_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        log2 = tf.math.log(tf.constant(2.0, dtype=x.dtype))
        return 2.0 * (log2 - x - tf.nn.softplus(-2.0 * x))
    

@tfp.distributions.RegisterKL(BoundedNormal, BoundedNormal)
def _kl_divergence_bounded_normal_dists(
    a: BoundedNormal, 
    b: BoundedNormal, 
    name: str = None
) -> tf.Tensor:
    return a._distribution.kl_divergence(b._distribution)
