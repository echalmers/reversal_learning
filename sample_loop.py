from agents.dqn import DQN
from agents.modulated_dqn import ModulatedDQN
from agents.comparators import MBMFParallel
from envs.multi_armed_bandit import VariableGamblingEnv
from envs.card_sorting import CardSortingEnv
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from copy import deepcopy
import random
from envs.wrappers import OneHotWrapper



# env = VariableGamblingEnv(p=[0.8, 0.1], rotation_interval=1000)

env = CardSortingEnv(n_classes=4, n_features=3, change_period=100)
env = OneHotWrapper(env, env.n_classes)

state, _ = env.reset()
print(env.unwrapped.card, state)

network = nn.Sequential(
    nn.Linear(len(state), 10),
    nn.Tanh(),
    nn.Linear(10, env.action_space.n)
)

mbmf_agent = MBMFParallel(
    mf_network=deepcopy(network),
    mb_network=nn.Sequential(nn.Linear(1, 10), nn.Tanh(), nn.Linear(10, env.unwrapped.n_features)),
    input_shape=state.shape,
    batch_size=10,
    replay_buffer_size=10,
    update_frequency=1,
    gamma=0,
    epsilon=0.1,
    lr=1e-2
)

standard_agent = DQN(
    network=deepcopy(network),
    input_shape=state.shape,
    batch_size=10,
    replay_buffer_size=10,
    update_frequency=1,
    gamma=0,
    epsilon=0.1,
    lr=1e-2
)

modulated_agent = ModulatedDQN(
    network=deepcopy(network),
    input_shape=state.shape,
    batch_size=10,
    replay_buffer_size=10,
    update_frequency=1,
    gamma=0,
    epsilon=0.1,
    lr=1e-2,
    softmax_temp=1,
)




for agent in [
    standard_agent,
    modulated_agent,
    mbmf_agent
]:
    np.random.seed(42)
    random.seed(42)

    STEPS = 5_000
    state, _ = env.reset()
    reward_history = np.zeros(STEPS)
    for step in range(STEPS):

        action = agent.select_action(state)
        new_state, reward, done, _, _ = env.step(action)

        agent.update(state=state, action=action, reward=reward, new_state=new_state, done=done)

        state = new_state

        reward_history[step] = reward

    plt.plot(reward_history.cumsum())
plt.show()
