import tensorflow as tf
import tensorflow_probability as tfp

from keras import layers
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations


class DenseCategorical(tfp.layers.DistributionLambda):

    def __init__(
        self,
        output_shape: int,
        activation: str = None,
        use_bias: bool = True,
        kernel_initializer: initializers.Initializer = None,
        bias_initializer: initializers.Initializer = None,
        kernel_regularizer: regularizers.Regularizer = None,
        bias_regularizer: regularizers.Regularizer = None,
        activity_regularizer: regularizers.Regularizer = None,
        kernel_constraint: constraints.Constraint = None,
        bias_constraint: constraints.Constraint = None,
        convert_to_tensor_fn: callable = tfp.distributions.Distribution.sample,
        dtype: tf.dtypes.DType = tf.int64,
        validate_args: bool = False,
        name: str = 'DenseCategorical',
        **kwargs
    ):
        output_shape = tf.TensorShape(output_shape)

        self._units = tf.math.reduce_prod(output_shape)
        self._activation = activations.get(activation)
        self._use_bias = use_bias

        if kernel_initializer is None:
            kernel_initializer = initializers.VarianceScaling(0.1)
        if bias_initializer is None:
            bias_initializer = initializers.Constant(0.0)
        
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._activity_regularizer = regularizers.get(activity_regularizer)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)
        
        super(DenseCategorical, self).__init__(
            make_distribution_fn=lambda inputs: DenseCategorical.new(
                inputs=inputs, 
                logits_layer=self.logits_layer, 
                output_shape=output_shape, 
                dtype=dtype, 
                validate_args=validate_args),
            convert_to_tensor_fn=convert_to_tensor_fn,
            name=name,
            **kwargs)

    def build(self, _) -> None:
        self.logits_layer = layers.Dense(
            units=self._units, 
            activation=self._activation,
            use_bias=self._use_bias,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name='logits_layer')

    @staticmethod
    def new(
        inputs: tf.Tensor|tuple[tf.Tensor, tf.Tensor], 
        logits_layer: layers.Layer,
        output_shape: tf.TensorShape,
        dtype: tf.dtypes.DType,
        validate_args: bool,
    ) -> tfp.distributions.Distribution:
        
        if isinstance(inputs, tf.Tensor):
            x = inputs 
            validity_mask = None
        else:
            x, validity_mask = inputs
            if validity_mask.dtype != tf.bool:
                validity_mask = tf.cast(validity_mask, tf.bool)

        logits = logits_layer(x)

        logits = tf.reshape(
            logits, tf.concat([tf.shape(logits)[:-1], output_shape], axis=0))

        if validity_mask is not None:
            logits = tf.where(validity_mask, logits, -1e10)

        distribution = tfp.distributions.Categorical(
            logits=logits, 
            dtype=dtype, 
            validate_args=validate_args)

        distribution = tfp.distributions.Independent(
            distribution=distribution,
            reinterpreted_batch_ndims=output_shape.ndims-1)
        
        return distribution
    