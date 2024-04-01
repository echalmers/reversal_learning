from agents.dqn import DQN
from agents.modulated_dqn import ModulatedDQN
from envs.card_sorting import CardSortingEnv
from torch import nn
from copy import deepcopy
import pandas as pd
from experiment import Experiment
import pickle
from envs.wrappers import OneHotWrapper


with open('../data/best_dqn_cardsorting.pkl', 'rb') as f:
    standard_agent = pickle.load(f)
with open('../data/best_modulated_cardsorting.pkl', 'rb') as f:
    modulated_agent = pickle.load(f)

results = []
n_features = [3, 4, 6, 8, 10, 12]
for rep in range(10):
    for n in n_features:
        env = CardSortingEnv(n_classes=n+1, n_features=n, change_period=200)
        env = OneHotWrapper(env, env.n_classes)
        state, _ = env.reset()

        network = nn.Sequential(
            nn.Linear(len(state), env.action_space.n)
        )

        df = Experiment(
            environment=env, n_steps=5_000,
            agent=DQN(
                network=deepcopy(network),
                input_shape=state.shape,
                batch_size=10,
                replay_buffer_size=10,
                update_frequency=1,
                gamma=0,
                epsilon=0.1,
                lr=standard_agent.lr,
            ),
        )()
        results.append(
            {'agent': 'TD learning', 'n_features': n, 'reward': df['reward'].mean()}
        )

        df = Experiment(
            environment=env, n_steps=5_000,
            agent=ModulatedDQN(
                network=deepcopy(network),
                input_shape=state.shape,
                batch_size=10,
                replay_buffer_size=10,
                update_frequency=1,
                gamma=0,
                epsilon=0.1,
                lr=modulated_agent.lr,
                softmax_temp=modulated_agent.softmax_temp,
            ),
        )()
        results.append(
            {'agent': 'new rule', 'n_features': n, 'reward': df['reward'].mean()}
        )

pd.DataFrame(results).to_csv('../data/results_cardsorting_paradox.csv', index=False)
