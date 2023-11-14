import tensorflow as tf

from functools import partial

from reinforceable.types import DType


class StreamingNormalizer(tf.Module):

    '''A streaming normalizer for nested and non-nested data.
    
    Args:
        clip_value:
            The clipping range applied to the normalized values. If set to 
            10, then the clipping range will be [-10, 10].
        center_mean:
            Whether to apply centering. In other words, whether to subtract
            the streamed mean values from the unnormalized values.
        dtype:
            The dtype to cast unnormalized values to. For instance, if the
            states are images of uint8, they will automatically be converted
            to dtype before being normalized. And they will not be converted 
            back to uint8. The dtype needs to be float, e.g. `tf.float32` or
            `tf.float64`. Default to `tf.float32`.
        name:
            The name of the module. Default to None.
    '''

    def __init__(
        self, 
        clip_value: float, 
        center_mean: bool, 
        dtype: DType = tf.float32,
        name: str = None
    ) -> None:
        super().__init__(name=name)
        self._dtype = dtype # TODO: assert dtype is float (?)
        self._built = False
        self.normalize_fn = partial(
            _normalize_tensors, 
            center_mean=center_mean, 
            clip_value=clip_value,
            dtype=self._dtype)

    def build(
        self, 
        inputs: tf.Tensor|tuple[tf.Tensor, ...], 
        reduce_ndims: int = 2
    ) -> None:
        input_shape = tf.nest.map_structure(lambda x: x.shape, inputs)
        inner_shape = tf.nest.map_structure(
            lambda s: tf.TensorShape(s)[reduce_ndims:], input_shape)
        with tf.init_scope():
            self.update_fn = partial(
                _update_moments, 
                reduce_ndims=reduce_ndims,
                dtype=self._dtype)
            with self.name_scope:
                self.mean = tf.nest.map_structure(
                    lambda s: tf.Variable(
                        initial_value=tf.zeros(s),
                        dtype=self._dtype,
                        trainable=False
                    ),
                    inner_shape
                )
                self.variance = tf.nest.map_structure(
                    lambda s: tf.Variable(
                        initial_value=tf.ones(s),
                        dtype=self._dtype,
                        trainable=False
                    ),
                    inner_shape
                )
                self.count = tf.nest.map_structure(
                    lambda s: tf.Variable(
                        initial_value=tf.keras.backend.epsilon(),
                        dtype=self._dtype,
                        trainable=False
                    ),
                    inner_shape
                )
            self._built = True

    def update(
        self, 
        inputs: tf.Tensor|tuple[tf.Tensor, ...], 
        reduce_ndims: int = 2
    ) -> None:
        if not self._built:
            self.build(inputs, reduce_ndims=reduce_ndims)
        tf.nest.map_structure(
            self.update_fn, inputs, self.mean, self.variance, self.count)

    def normalize(
        self, 
        inputs: tf.Tensor|tuple[tf.Tensor, ...]
    ) -> tf.Tensor|tuple[tf.Tensor, ...]:
        if self._built:
            return tf.nest.map_structure(
                self.normalize_fn, inputs, self.mean, self.variance)
        return inputs

    def reset(self) -> None:
        if self._built:
            eps = tf.keras.backend.epsilon()
            tf.nest.map_structure(
                lambda x: x.assign(tf.ones_like(x)), self.variance)
            tf.nest.map_structure(
                lambda x: x.assign(tf.zeros_like(x)), self.mean)
            tf.nest.map_structure(
                lambda x: x.assign(tf.zeros_like(x) + eps), self.count)


def _normalize_tensors(
    x: tf.Tensor, 
    mean: tf.Variable,
    variance: tf.Variable,
    center_mean: bool,
    clip_value: float,
    dtype: DType = tf.float32,
) -> tf.Tensor:
    x = tf.cast(x, dtype)
    if center_mean:
        x = x - mean 
    stddev = tf.maximum(tf.math.sqrt(variance), tf.keras.backend.epsilon())
    x = x / stddev
    return tf.clip_by_value(x, -clip_value, clip_value)

def _update_moments(
    x: tf.Tensor, 
    mean: tf.Variable,
    variance: tf.Variable,
    count: tf.Variable,
    reduce_ndims: tf.Tensor,
    dtype: DType = tf.float32,
) -> None:
    x = tf.cast(x, dtype)
    reduce_axes = tf.range(reduce_ndims)
    batch_mean = tf.math.reduce_mean(x, axis=reduce_axes)
    batch_variance = tf.math.reduce_variance(x, axis=reduce_axes)
    batch_count = tf.cast(
        tf.reduce_prod(tf.shape(x)[:reduce_ndims]), 
        dtype=count.dtype)

    delta = batch_mean - mean
    total_count = count + batch_count
    updated_mean = mean + delta * batch_count / total_count
    m_a = variance * count
    m_b = batch_variance * batch_count
    m2 = m_a + m_b + tf.math.square(delta) * count * batch_count / total_count
    updated_variance = m2 / total_count
    updated_count = total_count

    mean.assign(updated_mean)
    variance.assign(updated_variance)
    count.assign(updated_count)
