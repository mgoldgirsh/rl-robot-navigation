from collections import namedtuple

import numpy as np
import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import copy 
import matplotlib.pyplot as plt

from replay_buffer import Batch, ReplayMemory
from exp_schedule import ExponentialSchedule
from envs.generic_env import GenericWorld
from envs.obstacles_env import ObstaclesWorld

def rolling_average(data, *, window_size):
    """Smoothen the 1-d data array using a rollin average.

    Args:
        data: 1-d numpy.array
        window_size: size of the smoothing window

    Returns:
        smooth_data: a 1-d numpy.array with the same size as data
    """
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, *, num_layers=3, hidden_dim=256):
        """Deep Q-Network PyTorch model.

        Args:
            - state_dim: Dimensionality of states
            - action_dim: Dimensionality of actions
            - num_layers: Number of total linear layers
            - hidden_dim: Number of neurons in the hidden layers
        """

        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # YOUR CODE HERE:  define the layers of your model such that
        # * there are `num_layers` nn.Linear modules / layers
        # * all activations except the last should be ReLU activations
        #   (this can be achieved either using a nn.ReLU() object or the nn.functional.relu() method)
        # * the last activation can either be missing, or you can use nn.Identity()
        self.layers = [(nn.Linear(self.state_dim, hidden_dim)), nn.ReLU()]

        for _ in range(num_layers-2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())

        # last one
        self.layers.append(nn.Linear(hidden_dim, self.action_dim))

        self.linear_stack = nn.Sequential(*self.layers)
        
    def forward(self, states) -> torch.Tensor:
        """Q function mapping from states to action-values.

        :param states: (*, S) torch.Tensor where * is any number of additional
                dimensions, and S is the dimensionality of state-space.
        :rtype: (*, A) torch.Tensor where * is the same number of additional
                dimensions as the `states`, and A is the dimensionality of the
                action-space.  This represents the Q values Q(s, .).
        """
        # YOUR CODE HERE:  use the defined layers and activations to compute
        # the action-values tensor associated with the input states.
        
        result = copy.deepcopy(states)
        for layer in self.layers:
            result = layer(result)
        return result
    
    
    # utility methods for cloning and storing models.  DO NOT EDIT
    @classmethod
    def custom_load(cls, data):
        model = cls(*data['args'], **data['kwargs'])
        model.load_state_dict(data['state_dict'])
        return model

    def custom_dump(self):
        return {
            'args': (self.state_dim, self.action_dim),
            'kwargs': {
                'num_layers': self.num_layers,
                'hidden_dim': self.hidden_dim,
            },
            'state_dict': self.state_dict(),
        }


# test code, do not edit


def _test_dqn_forward(dqn_model, input_shape, output_shape):
    """Tests that the dqn returns the correctly shaped tensors."""
    inputs = torch.torch.randn((input_shape))
    outputs = dqn_model(inputs)

    if not isinstance(outputs, torch.FloatTensor):
        raise Exception(
            f'DQN.forward returned type {type(outputs)} instead of torch.Tensor'
        )

    if outputs.shape != output_shape:
        raise Exception(
            f'DQN.forward returned tensor with shape {outputs.shape} instead of {output_shape}'
        )

    if not outputs.requires_grad:
        raise Exception(
            f'DQN.forward returned tensor which does not require a gradient (but it should)'
        )


dqn_model = DQN(10, 4)
_test_dqn_forward(dqn_model, (64, 10), (64, 4))
_test_dqn_forward(dqn_model, (2, 3, 10), (2, 3, 4))
del dqn_model

dqn_model = DQN(64, 16)
_test_dqn_forward(dqn_model, (64, 64), (64, 16))
_test_dqn_forward(dqn_model, (2, 3, 64), (2, 3, 16))
del dqn_model

# testing custom dump / load
dqn1 = DQN(10, 4, num_layers=10, hidden_dim=20)
dqn2 = DQN.custom_load(dqn1.custom_dump())
assert dqn2.state_dim == 10
assert dqn2.action_dim == 4
assert dqn2.num_layers == 10
assert dqn2.hidden_dim == 20

def train_dqn_batch(optimizer: torch.optim.Optimizer, batch: Batch, dqn_model: nn.Module, dqn_target: nn.Module, gamma: float) -> float:
    """Perform a single batch-update step on the given DQN model.

    :param optimizer: nn.optim.Optimizer instance.
    :param batch:  Batch of experiences (class defined earlier).
    :param dqn_model:  The DQN model to be trained.
    :param dqn_target:  The target DQN model, ~NOT~ to be trained.
    :param gamma:  The discount factor.
    :rtype: float  The scalar loss associated with this batch.
    """
    # YOUR CODE HERE:  compute the values and target_values tensors using the
    # given models and the batch of data.
     
    values : torch.Tensor = dqn_model(batch.states).gather(1, batch.actions)
    # target_values = batch.rewards + gamma * torch.max(dqn_target(batch.next_states)).detach()
    # target_values = batch.rewards + gamma * dqn_target(batch.next_states).max(1)[0].detach().unsqueeze(1)
    torch
    # Compute the target values tensor
    with torch.no_grad():
        next_state_values = torch.zeros(len(batch.next_states))
        non_final_mask = torch.tensor(tuple(map(lambda s: s.item() is False, batch.dones)), dtype=torch.bool)
        next_state_values[non_final_mask] = dqn_target(batch.next_states[non_final_mask]).max(1)[0].detach()

        target_values = batch.rewards + gamma * next_state_values.unsqueeze(1)
    
    # DO NOT EDIT FURTHER

    assert (
        values.shape == target_values.shape
    ), 'Shapes of values tensor and target_values tensor do not match.'

    # testing that the value tensor requires a gradient,
    # and the target_values tensor does not
    assert values.requires_grad, 'values tensor should not require gradients'
    assert (
        not target_values.requires_grad
    ), 'target_values tensor should require gradients'

    # computing the scalar MSE loss between computed values and the TD-target
    loss = F.mse_loss(values, target_values)

    optimizer.zero_grad()  # reset all previous gradients
    loss.backward()  # compute new gradients
    optimizer.step()  # perform one gradient descent step

    return loss.item()


def train_dqn(
    env : GenericWorld,
    observation_space, 
    action_space,
    num_steps,
    *,
    num_saves=5,
    replay_size,
    replay_prepopulate_steps=0,
    batch_size,
    exploration : ExponentialSchedule,
    gamma : float,
    render: bool = False
):
    """
    DQN algorithm.

    Compared to previous training procedures, we will train for a given number
    of time-steps rather than a given number of episodes.  The number of
    time-steps will be in the range of millions, which still results in many
    episodes being executed.

    Args:
        - env: The openai Gym environment
        - num_steps: Total number of steps to be used for training
        - num_saves: How many models to save to analyze the training progress.
        - replay_size: Maximum size of the ReplayMemory
        - replay_prepopulate_steps: Number of steps with which to prepopulate
                                    the memory
        - batch_size: Number of experiences in a batch
        - exploration: a ExponentialSchedule
        - gamma: The discount factor

    Returns: (saved_models, returns)
        - saved_models: Dictionary whose values are trained DQN models
        - returns: Numpy array containing the return of each training episode
        - lengths: Numpy array containing the length of each training episode
        - losses: Numpy array containing the loss of each training batch
    """

    # get the state_size from the environment
    state_size = observation_space

    # initialize the DQN and DQN-target models
    dqn_model = DQN(state_size, action_space)
    dqn_target = DQN.custom_load(dqn_model.custom_dump())

    # initialize the optimizer
    optimizer = torch.optim.Adam(dqn_model.parameters())

    # initialize the replay memory and prepopulate it
    memory = ReplayMemory(replay_size, state_size)
    memory.populate(env, replay_prepopulate_steps)

    # initiate lists to store returns, lengths and losses
    rewards = []
    returns = []
    lengths = []
    losses = []
    
    G = 0 # discounted returns

    # initiate structures to store the models at different stages of training
    t_saves = np.linspace(0, num_steps, num_saves - 1, endpoint=False)
    saved_models = {}

    i_episode = 0  # use this to indicate the index of the current episode
    t_episode = 0  # use this to indicate the time-step inside current episode

    state = env.reset(render=True)  # initialize state of first episode

    # iterate for a total of `num_steps` steps
    pbar = tqdm.trange(num_steps, ncols=100)
    for t_total in pbar:
        # use t_total to indicate the time-step from the beginning of training

        # save model
        if t_total in t_saves:
            model_name = f'{100 * t_total / num_steps:04.1f}'.replace('.', '_')
            saved_models[model_name] = copy.deepcopy(dqn_model)

        # YOUR CODE HERE:
        #  * sample an action from the DQN using epsilon-greedy
        #  * use the action to advance the environment by one step
        #  * store the transition into the replay memory
        
        eps = exploration.value(t_total)
        action = None
        if (np.random.random() < eps):
            action = np.random.randint(action_space)
            #action = env.action_space.sample()
        else:
            action = torch.argmax(dqn_model(torch.Tensor(state)), dim=0).item()
        
        next_state, reward, done = env.step(action, render=render)
        rewards.append(reward)
        
        G = reward + gamma * G
        memory.add(copy.deepcopy(state), action, reward, copy.deepcopy(next_state), done)
        state = next_state
        
        # YOUR CODE HERE:  once every 4 steps,
        #  * sample a batch from the replay memory
        #  * perform a batch update (use the train_dqn_batch() method!)
        
        if (t_total % 4 == 0):
            sampled_batch = memory.sample(batch_size)
            loss = train_dqn_batch(optimizer, sampled_batch, dqn_model, dqn_target, gamma)
            losses.append(loss)

        # YOUR CODE HERE:  once every 10_000 steps,
        #  * update the target network (use the dqn_model.state_dict() and
        #    dqn_target.load_state_dict() methods!)
        
        if (t_total % 10_000 == 0):
            dqn_target.load_state_dict(dqn_model.state_dict())

        if done:
            # YOUR CODE HERE:  anything you need to do at the end of an
            # episode, e.g. compute return G, store stuff, reset variables,
            # indices, lists, etc.

            state = env.reset(render=True)
            lengths.append(t_episode + 1)
            returns.append(G)
            
            pbar.set_description(
                f'Episode: {i_episode} | Steps: {t_episode + 1} | Return: {G:5.2f} | Epsilon: {eps:4.2f}'
            )
            
            t_episode = 0
            i_episode += 1
            G = 0
            rewards = []
        else:
            # YOUR CODE HERE:  anything you need to do within an episode
            t_episode += 1

    saved_models['100_0'] = copy.deepcopy(dqn_model)

    return (
        saved_models,
        np.array(returns),
        np.array(lengths),
        np.array(losses),
    )

def plot(returns, lengths, losses):
    ### YOUR PLOTTING CODE HERE
    plt.plot(range(len(returns)), returns, color='red', alpha=0.5, label='raw return data')
    plt.plot(range(len(returns)), rolling_average(data=returns, window_size=100), color='red', label='smooth return data')
    plt.title("MountainCar: Return per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Discounted Return")
    plt.legend()
    plt.show()

    plt.plot(range(len(losses)), losses, color='blue', alpha=0.5, label='raw loss data')
    plt.plot(range(len(losses)), rolling_average(data=losses, window_size=100), color='blue', label='smooth loss data')
    plt.title("MountainCar: Loss per 4 Steps")
    plt.xlabel("4 Steps")
    plt.ylabel("Loss value")
    plt.legend()
    plt.show()

    plt.plot(range(len(lengths)), lengths, color='green', alpha=0.5, label='raw length data')
    plt.plot(range(len(lengths)), rolling_average(data=lengths, window_size=100), color='green', label='smooth length data')
    plt.title("MountainCar: Epsiode Length per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    env = ObstaclesWorld(500, 500, see_all=True)
    gamma = 0.99

    # we train for many time-steps;  as usual, you can decrease this during development / debugging.
    # but make sure to restore it to 1_500_000 before submitting.
    num_steps = 500_000
    num_saves = 5  # save models at 0%, 25%, 50%, 75% and 100% of training

    replay_size = 200_000
    replay_prepopulate_steps = 0 #10 #50_000

    batch_size = 64
    exploration = ExponentialSchedule(1.0, 0.01, 300_000)

    # this should take about 90-120 minutes on a generic 4-core laptop
    dqn_models, returns, lengths, losses = train_dqn(
        env,
        observation_space=184,
        action_space=2,
        num_steps=num_steps,
        num_saves=num_saves,
        replay_size=replay_size,
        replay_prepopulate_steps=replay_prepopulate_steps,
        batch_size=batch_size,
        exploration=exploration,
        gamma=gamma,
        render=True
    )

    plot()
    # assert len(dqn_models) == num_saves
    # assert all(isinstance(value, DQN) for value in dqn_models.values())

    # saving computed models to disk, so that we can load and visualize them later.
    # checkpoint = {key: dqn.custom_dump() for key, dqn in dqn_models.items()}
    # torch.save(checkpoint, f'checkpoint_{env.spec.id}.pt')