# Reinforceable

## Status
Work in progress (unfinished). Any feedback is welcomed.

## About
- Deep reinforcement learning (RL) implementations using [TF's probabilistic library](https://www.tensorflow.org/probability), specifically focusing on agents using recurrent neural networks (RNNs).

- A light-weight package, only dependent on TensorFlow, TensorFlow probability and Gymnasium. Although possibly subject to change in the future, the RL implementations are currently implemented and optimized for a non-distributed setup (i.e., for a single CPU and/or GPU setup). 

> A quick benchmark of the recurrent PPO algorithm in the [Atari](https://gymnasium.farama.org/environments/atari/) environments (using a single processor and GPU, and 32 parallel environments), shows that it processes, roughly, 6-12M frames per hour &#151; approximately 1700-3300 frames per second (FPS).  

## Highlights 

- Python RL environments (e.g., [Gym(nasium)](https://github.com/Farama-Foundation/Gymnasium) enviroments such as [Classic Control](https://gymnasium.farama.org/environments/classic_control/) and [Atari](https://gymnasium.farama.org/environments/atari/) environments) can be run on the TF graph, allowing the complete interaction loop (agent-environment interaction) to run non-eagerly. See [Driver](https://github.com/akensert/reinforceable/blob/main/reinforceable/driver.py).

- A PPO algorithm that deals with *partial observability* is implemented ([RecurrentPPOAgent](https://github.com/akensert/reinforceable/blob/main/reinforceable/ppo/ppo_agent.py)). [RecurrentPPOAgent](https://github.com/akensert/reinforceable/blob/main/reinforceable/ppo/ppo_agent.py) makes use of stateful RNNs to pass hidden states between time steps, allowing the agent to make decisions based on past states as well as the current state (Figure 1A). This contrasts to a typical PPO implementations wherein the agent makes decisions based on the current state only (Figure 1B).

<img src="https://github.com/akensert/reinforceable/blob/main/media/ppo.jpg" alt="PPO" width="800">

> The use of hidden states is a clever way to pass experiences through time. One limitation of this approach however, is that the hidden states correspond to incomplete trajectories (chunks of trajectories) for each training iteration &#151; a limitation especially emphasized for longer episodes and off-policy RL (using experience replay). For further reading, see [R2D2 paper](https://openreview.net/pdf?id=r1lyTjAqYX).

## Implementations

- Agents
    - [RecurrentPPOAgent](https://github.com/akensert/reinforceable/blob/main/reinforceable/ppo/ppo_agent.py)
- Layers
    - [DenseNormal](https://github.com/akensert/reinforceable/blob/main/reinforceable/layers/dense_normal.py)  - for continuous actions.
    - [DenseCategorical](https://github.com/akensert/reinforceable/blob/main/reinforceable/layers/dense_categorical.py) - for categorical actions.
    - [DenseBernoulli](https://github.com/akensert/reinforceable/blob/main/reinforceable/layers/dense_bernoulli.py) - for binary actions.
    - [StatefulRNN](https://github.com/akensert/reinforceable/blob/main/reinforceable/layers/stateful_rnn.py) - for passing information between states
- Distributions
    - [BoundedNormal](https://github.com/akensert/reinforceable/blob/main/reinforceable/distributions/bounded_normal.py) - a bounded normal distribution, inheriting from `TransformedDistribution`.
- Environments
    - [TimestepEnv](https://github.com/akensert/reinforceable/blob/main/reinforceable/envs/gym_wrappers.py) - wraps [gymnasium](https://gymnasium.farama.org/) environments to output [Timesteps](https://github.com/akensert/reinforceable/blob/main/timestep.py). (May get deprecated in the future, but for now used to deal with [tf.numpy_function](https://www.tensorflow.org/api_docs/python/tf/numpy_function).)
    - [AsyncEnvironment](https://github.com/akensert/reinforceable/blob/main/reinforceable/envs/async_env.py) - allowing multiple independent (fow now [TimestepEnv](https://github.com/akensert/reinforceable/blob/main/reinforceable/envs/gym_wrappers.py)) environments to run in parallel. 

For hybrid action spaces, just combine action layers:
```python
from keras import Model
from reinforceable import layers
# ... 
action_1 = layers.DenseNormal((2,), [-1., 1.], x)   # continuous action, dim=2
action_2 = layers.DenseCategorical((10,))(x)        # discrete action, n=10
policy_network = Model(inputs, (action_1, action_2))
# ...
```

## Examples

See [examples/example.ipynb](https://github.com/akensert/reinforceable/blob/main/examples/example.ipynb).

## Dependencies
- Python >= 3.10
    - tensorflow >= 2.13.0
    - tensorflow-probability >= 0.20.1
    - gymnasium[all] >= 0.26.2

> For atari environments, atari ROMs need to be installed. [See here](https://gymnasium.farama.org/environments/atari/).

## Installation
With SSH:
```
git clone git@github.com:akensert/reinforceable.git
pip install -e .
```
With HTTPS:
```
git clone https://github.com/akensert/reinforceable.git
pip install -e .
```

