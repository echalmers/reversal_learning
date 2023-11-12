from agents.dqn import DQN
from agents.modulated_dqn import ModulatedDQN
from agents.comparators import MBMFParallelOptimal, MBMFParallel2
from envs.multi_armed_bandit import VariableGamblingEnv
from envs.card_sorting import CardSortingEnv
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from copy import deepcopy
import pandas as pd
from envs.wrappers import OneHotWrapper



env = CardSortingEnv(n_classes=4, n_features=3, change_period=100)
env = OneHotWrapper(env, env.n_classes)

state, _ = env.reset()

network = nn.Sequential(
    nn.Linear(len(state), 10),
    nn.Tanh(),
    nn.Linear(10, env.action_space.n)
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
    softmax_temp=0.5,
)

mbmf_agent_optimal = MBMFParallelOptimal(
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

mbmf_agent_2 = MBMFParallel2(
    DQN(
        network=nn.Sequential(nn.Linear(len(state), 10), nn.Tanh(), nn.Linear(10, env.unwrapped.n_features + env.action_space.n)),
        input_shape=state.shape,
        batch_size=10,
        replay_buffer_size=10,
        update_frequency=1,
        gamma=0,
        epsilon=0.1,
        lr=1e-2
    ),
    n_classes=env.unwrapped.n_classes,
    n_features=env.unwrapped.n_features
)

agents = [
    modulated_agent,
    standard_agent,
    mbmf_agent_2
]

results = []

for agent in agents:
    # np.random.seed(42)
    # random.seed(42)

    STEPS = 10_000
    state, _ = env.reset()
    reward_history = np.zeros(STEPS)
    for step in range(STEPS):

        correct_action = env.get_correct_action()  # just for results - agent can't see it
        action = agent.select_action(state)
        new_state, reward, done, _, _ = env.step(action)
        results.append({'agent': agent.__class__.__name__, 'step': step, 'correct_action': correct_action, 'action': action, 'reward': reward})

        reward_history[step] = reward

        agent.update(state=state, action=action, reward=reward, new_state=new_state, done=done)

        state = new_state

    plt.plot(reward_history.cumsum())

pd.DataFrame(results).to_csv('results_cardsorting.csv', index=False)

plt.legend([x.__class__.__name__ for x in agents])
plt.show()
