from agents.dqn import DQN
from agents.modulated_dqn import ModulatedDQN
from agents.comparators import MBMFParallelOptimal, MBMFParallel2
from envs.multi_armed_bandit import VariableGamblingEnv
from agents.tabular import NewLearner
from envs.card_sorting import CardSortingEnv
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from copy import deepcopy
import pandas as pd
from experiment import Experiment
import pickle
import seaborn as sns
import os


if not os.path.isfile('../data/results_bandit_paradox.csv'):

    with open('../data/best_perfect_info_bandit.pkl', 'rb') as f:
        perfect_info = pickle.load(f)
    with open('../data/best_modulated_tabular_bandit.pkl', 'rb') as f:
        modulated_tabular = pickle.load(f)
    with open('../data/best_q_learner_bandit.pkl', 'rb') as f:
        q_learner = pickle.load(f)

    results = []
    ns = [3, 4, 5, 6, 7, 8, 9, 10]
    for rep in range(25):
        for agent in [modulated_tabular, q_learner]:
            for n in ns:
                p = np.random.rand(n) * 0.5 + 0.25
                p[-1] = 0
                p[0] = 0.9
                env = VariableGamblingEnv(p=p, rotation_interval=100)
                state, _ = env.reset()

                this_agent = agent.__class__(
                    action_list=np.arange(len(p)), gamma=agent.gamma, alpha=agent.alpha,
                    default_value=agent.default_value, softmax_temperature=agent.softmax_temperature,
                )

                df = Experiment(
                    environment=env, agent=this_agent, n_steps=1000,
                # params={'action_list': [np.arange(len(p))], 'gamma': [0.75], 'alpha': [2, 1, 0.5, 0.1], 'default_value': [1], 'softmax_temperature': [0.25, 1], 'softmax_temperature_value_update': [tau]},
                # param_search_steps=1_000
                )()
                results.append(
                    {'agent': agent.__class__.__name__, 'n': n, 'reward': df['reward'].mean()}
                )

    pd.DataFrame(results).to_csv('../data/results_bandit_paradox.csv', index=False)


results = pd.read_csv('../data/results_bandit_paradox.csv', index_col=None)
sns.lineplot(data=results, x='n', y='reward', hue='agent')
plt.show()