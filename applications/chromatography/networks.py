import tensorflow as tf
import keras 
import functools 

from keras.layers import *
from keras.src.engine import keras_tensor

from reinforceable.layers import *


Conv1D = functools.partial(Conv1D, padding='same', activation='relu')


def EncoderNetwork(
    batch_size: int,
    chromatogram_shape: list[int],
    phi_shape: list[int],
    recurrent: bool = False,
) -> keras.Model:
    
    chromatogram = Input((batch_size,) + chromatogram_shape, dtype=tf.float32)
    phi_target = Input((batch_size,) + phi_shape, dtype=tf.float32)

    # If recurrent == False, hidden_states_mask is unused.
    hidden_state_mask = Input((batch_size,) + (1,), dtype=tf.bool)

    x = chromatogram 
    
    # Note: TimeDistributed is used because a time dimension is added in 
    #       addition to a batch dimension. This is necessary as hidden states
    #       are passed along the trajectory, via recurrent neural nets.
    #
    #       If a PPOAgent is used instead (so no hidden states), the time 
    #       dimension as well as TimeDistributed cam both be omitted.

    x = TimeDistributed(Conv1D(32, 9, strides=4))(x)
    x = TimeDistributed(Conv1D(32, 9, strides=4))(x)
    x = TimeDistributed(Conv1D(64,  7, strides=4))(x)
    x = TimeDistributed(Conv1D(64,  5, strides=2))(x)
    x = TimeDistributed(Conv1D(128, 3, strides=2))(x)
    x = TimeDistributed(Conv1D(128, 3, strides=2))(x)

    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(128))(x)

    x_aux = TimeDistributed(Dense(128))(phi_target)

    # TODO: How to combine x and x_aux? I.e., how to inform the agent
    #       what phi-program was used (that resulted in the chromatogram)
    x = Add()([x, x_aux]) 
    
    if recurrent:
        x = StatefulRNN(LSTMCell(128))(x, hidden_state_mask)

    return keras.Model(
        inputs=(
            {
                'chromatogram': chromatogram, 
                'phi_target': phi_target
            },
            hidden_state_mask
        ),
        outputs=x,
        name='EncoderNetwork'
    )

def PolicyNetwork(
    inputs: keras_tensor.KerasTensor,
    action_shape: list[int],
) -> keras.Model:
    x = TimeDistributed(Dense(units=128, activation='relu'))(inputs)
    distrib = DenseNormal(
        action_shape, [-1., 1.], 
        scale_bias_initializer=keras.initializers.Constant(-1.0),
        state_dependent_scale=True)(x)
    return keras.Model(inputs, distrib, name='PolicyNetwork')

def ValueNetwork(
    inputs: keras_tensor.KerasTensor
) -> keras.Model:
    x = TimeDistributed(Dense(128, activation='relu'))(inputs)
    value = TimeDistributed(Dense(1))(x)
    return keras.Model(inputs, value, name='ValueNetwork')

