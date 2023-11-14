# Different types used with Reinforceable. 
#
# These types are used as typing hints for the remaining modules of 
# Reinforceable, and hopefully improves the readability of these modules.


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import typing


class Spec(typing.Protocol):

    def shape(self):
        ...

    def dtype(self):
        ...


class GymEnvironment(typing.Protocol):

    def step(self, action):
        ...

    def reset(self):
        ...

    def render(self):
        ...

    def seed(self):
        ...

    # etc.

    
GymEnvironmentConstructor = typing.Type[GymEnvironment]

Shape = typing.Union[tf.TensorShape, tuple[int, ...], list[int]]
DType = typing.Union[tf.DType, np.dtype, str]

# More accurately 'TensorLike', but prefer to simplify the name.
# Each type can be converted to a tf.Tensor via tf.convert_to_tensor
Tensor = typing.Union[
    tf.Tensor,
    np.ndarray, 
    list,
    tuple,
    int, 
    float, 
    str, 
    bool
]

TensorOrSpec = typing.Union[Tensor, Spec]
Distribution = tfp.distributions.Distribution
TensorArray = tf.TensorArray

Tnest = typing.TypeVar('Tnest')
Trecursive = typing.TypeVar('Trecursive')
Nested = typing.Union[
    Tnest, 
    typing.Iterable[Trecursive], 
    typing.Mapping[typing.Text, Trecursive]
]
NestedShape = Nested[Shape, 'NestedShape']
NestedDType = Nested[DType, 'NestedDType']
NestedTensor = Nested[Tensor, 'NestedTensor']
NestedTensorArray = Nested[tf.TensorArray, 'NestedTensorArray']
NestedSpec = Nested[Spec, 'NestedSpec']
NestedTensorOrSpec = typing.Union[NestedTensor, NestedSpec]
NestedDistribution = Nested[Distribution, 'NestedDistribution']

FlatTimestep = typing.List[Tensor]

TrainData = typing.Union[
    typing.NamedTuple, 
    typing.Tuple[typing.NamedTuple, Tensor] # with sample weights
]
TrainInfo = typing.Dict[str, Tensor]