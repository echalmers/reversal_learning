from agents.dqn import DQN
from agents.modulated_dqn import ModulatedDQN
from agents.comparators import MBMFParallelOptimal, MBMFParallel2
from envs.multi_armed_bandit import VariableGamblingEnv
from agents.tabular import QLearner, NewLearner as ModulatedTabularLearner
from envs.card_sorting import CardSortingEnv
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from copy import deepcopy
from agents.bandit_baselines import PerfectInfoAgent, GradientLearner
from experiment import optimize_params
import pickle

if __name__ == '__main__':
    batch_size = 5
    n = 7
    p = np.random.rand(n) * 0.5 + 0.25
    p[-1] = 0
    p[0] = 0.9
    rotation_interval = 100
    env = VariableGamblingEnv(p=p, rotation_interval=rotation_interval)
    state, _ = env.reset()


    best_perfect_info = optimize_params(environment=env, agent_class=PerfectInfoAgent,
                                        params={'n': [len(p)], 'c': [0.25, 0.5, 1, 2, 4], 'rotation_interval': [rotation_interval]},
                                        param_search_steps=10000
                                        )
    with open('../data/best_perfect_info_bandit.pkl', 'wb') as f:
        pickle.dump(best_perfect_info, f)

    best_modulated_tabular = optimize_params(environment=env, agent_class=ModulatedTabularLearner,
                                     params={'action_list': [np.arange(len(p))], 'gamma': [0.75], 'alpha': [1, 0.5, 0.1], 'default_value': [1], 'softmax_temperature': [0.25, 1]},
                                     param_search_steps=10_000
                                     )
    with open('../data/best_modulated_tabular_bandit.pkl', 'wb') as f:
        pickle.dump(best_modulated_tabular, f)

    best_q_learner = optimize_params(environment=env, agent_class=QLearner,
                                     params={'action_list': [np.arange(len(p))], 'gamma': [0.75], 'alpha': [1, 0.5, 0.1], 'default_value': [1], 'softmax_temperature': [0.25, 1]},
                                     param_search_steps=10_000
                                     )
    with open('../data/best_q_learner_bandit.pkl', 'wb') as f:
        pickle.dump(best_q_learner, f)

    best_gradient_agent = optimize_params(environment=env, agent_class=GradientLearner,
                                          params={'n': [len(p)], 'alpha': [1, 0.5], 'baseline': [True, False]},
                                          param_search_steps=10_000
                                          )
    with open('../data/best_gradient_agent.pkl', 'wb') as f:
        pickle.dump(best_gradient_agent, f)
