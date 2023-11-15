import tensorflow as tf
import tensorflow_probability as tfp

from keras import layers
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations

from reinforceable.distributions.bounded_normal import BoundedNormal


class DenseNormal(tfp.layers.DistributionLambda):

    def __init__(
        self,
        output_shape,
        output_bounds,
        hard_bounds: bool = False,
        state_dependent_scale: bool = False,
        loc_activation: str = None,
        scale_activation: str = 'softplus',
        use_bias: bool = True,
        kernel_initializer: initializers.Initializer = None,
        bias_initializer: initializers.Initializer = None,
        kernel_regularizer: regularizers.Regularizer = None,
        bias_regularizer: regularizers.Regularizer = None,
        activity_regularizer: regularizers.Regularizer = None,
        kernel_constraint: constraints.Constraint = None,
        bias_constraint: constraints.Constraint = None,
        convert_to_tensor_fn: callable = tfp.distributions.Distribution.sample,
        validate_args: bool = False,
        name: str = 'DenseNormal',
        **kwargs
    ):
        output_shape = tf.TensorShape(output_shape)
        output_bounds = tf.convert_to_tensor(output_bounds)
        output_bounds = tf.broadcast_to(
            output_bounds, output_shape.concatenate([2,]))

        if output_shape != output_bounds.shape[:-1]:
            raise ValueError('`output_shape` != `output_bounds.shape[:-1]`')

        self._state_dependent_scale = state_dependent_scale

        self._units = tf.math.reduce_prod(output_shape)
        self._loc_activation = activations.get(loc_activation)
        self._scale_activation = activations.get(scale_activation)
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


        super(DenseNormal, self).__init__(
            make_distribution_fn=lambda inputs: DenseNormal.new(
                inputs, 
                loc_layer=self.loc_layer, 
                scale_layer=self.scale_layer, 
                output_bounds=output_bounds,
                hard_bounds=hard_bounds, 
                output_shape=output_shape,
                validate_args=validate_args,
            ),
            convert_to_tensor_fn=convert_to_tensor_fn,
            name=name,
            **kwargs)

    def build(self, _) -> None:
        if self._state_dependent_scale:
            self.scale_layer = layers.Dense(
                units=self._units, 
                activation=self._scale_activation,
                use_bias=self._use_bias,
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                activity_regularizer=self._activity_regularizer,
                kernel_constraint=self._kernel_constraint,
                bias_constraint=self._bias_constraint,
                name='scale_layer')
        else:
            self.scale_layer = _BiasLayer(
                dim=self._units,
                activation=self._scale_activation,
                bias_initializer=self._bias_initializer,
                bias_regularizer=self._bias_regularizer,
                bias_constraint=self._bias_constraint,
                name='scale_layer')

        self.loc_layer = layers.Dense(
            units=self._units, 
            activation=self._loc_activation,
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
        loc_layer: layers.Layer, 
        scale_layer: layers.Layer, 
        output_bounds: tf.Tensor,
        hard_bounds: bool,
        output_shape: tf.TensorShape, 
        validate_args: bool = False, 
    ) -> tfp.distributions.Distribution:
        
        if isinstance(inputs, tf.Tensor):
            x = inputs 
            validity_mask = None
        else:
            x, validity_mask = inputs
            validity_mask = tf.cast(validity_mask, tf.bool)

        x_loc = loc_layer(x)
        x_scale = scale_layer(x)

        outer_shape = tf.shape(x_loc)[:-1]
        inner_shape = output_shape

        x_loc = tf.reshape(
            x_loc, tf.concat([outer_shape, inner_shape], axis=0))
        x_scale = tf.reshape(
            x_scale, tf.concat([outer_shape, inner_shape], axis=0))

        if not hard_bounds:
            x_loc = DenseNormal.rescale_loc(x_loc, output_bounds)

        distribution = tfp.distributions.Normal(
            loc=x_loc, scale=x_scale, validate_args=validate_args)
        
        if validity_mask is not None:
            distribution = tfp.distributions.Masked(
                distribution, validity_mask=validity_mask)

        distribution = tfp.distributions.Independent(
            distribution, reinterpreted_batch_ndims=output_shape.ndims)

        if not hard_bounds:
            return distribution

        return BoundedNormal(
            distribution, output_bounds, validate_args)

    @staticmethod
    def rescale_loc(
        loc: tf.Tensor, 
        bounds: tf.Tensor, 
    ) -> tf.Tensor: 
        means = (bounds[..., 1] + bounds[..., 0]) / 2.0
        magnitudes = (bounds[..., 1] - bounds[..., 0]) / 2.0
        return means + tf.math.tanh(loc) * magnitudes


class _BiasLayer(layers.Layer):

    def __init__(
        self, 
        dim: int,
        activation: layers.Activation,
        bias_initializer: initializers.Initializer, 
        bias_regularizer: regularizers.Regularizer, 
        bias_constraint: constraints.Constraint, 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._dim = dim
        self._activation = activations.get(activation)
        self._bias_initializer = initializers.get(bias_initializer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._bias_constraint = constraints.get(bias_constraint)

    def build(self, _) -> None:
        self.bias = self.add_weight(
            name='bias',
            shape=(self._dim,),
            initializer=self._bias_initializer,
            dtype=self.dtype,
            trainable=True)
        self.built = True

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.zeros(
            shape=tf.concat([
                tf.shape(inputs)[:-1], tf.shape(self.bias)], axis=0),
            dtype=inputs.dtype)
        return self._activation(tf.nn.bias_add(inputs, self.bias))

    def compute_output_shape(
        self, 
        input_shape: tf.TensorShape
    ) -> tf.TensorShape:
        return input_shape

    def get_config(self) -> dict:
        config = {
            'dim': self._dim,
            'activation': activations.serialize(self._activation),
            'bias_initializer': initializers.serialize(self._bias_initializer),
            'bias_regularizer': initializers.serialize(self._bias_regularizer),
            'bias_constraint': initializers.serialize(self._bias_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

