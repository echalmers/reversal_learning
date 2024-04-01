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


n = 7

with open('../data/best_perfect_info_bandit.pkl', 'rb') as f:
    perfect_info = pickle.load(f)
with open('../data/best_modulated_tabular_bandit.pkl', 'rb') as f:
    modulated_tabular = pickle.load(f)
with open('../data/best_q_learner_bandit.pkl', 'rb') as f:
    q_learner = pickle.load(f)

sweep_results = []
taus = [0.01, 0.1, 1, 10, 100, 1000]
for rep in range(25):
    for tau in taus:
        p = np.random.rand(n) * 0.5 + 0.25
        p[-1] = 0
        p[0] = 0.9
        env = VariableGamblingEnv(p=p, rotation_interval=100)
        state, _ = env.reset()

        new_rule_agent = deepcopy(modulated_tabular)
        new_rule_agent.softmax_temperature_value_update = tau
        new_rule_agent.softmax_temperature = q_learner.softmax_temperature
        new_rule_agent.alpha = q_learner.alpha * 1
        df = Experiment(
            environment=env, agent=new_rule_agent, n_steps=1000,
            # params={'action_list': [np.arange(len(p))], 'gamma': [0.75], 'alpha': [2, 1, 0.5, 0.1], 'default_value': [1], 'softmax_temperature': [0.25, 1], 'softmax_temperature_value_update': [tau]},
            # param_search_steps=1_000
        )()
        sweep_results.append(
            {'tau': tau, 'reward': df['reward'].mean()}
        )

pd.DataFrame(sweep_results).to_csv('../data/results_bandit_sweep.csv', index=False)
