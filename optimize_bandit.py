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
from agents.bandit_baselines import PerfectInfoAgent
from experiment import optimize_params
import pickle


batch_size = 5
n = 7
p = np.random.rand(n) * 0.5 + 0.25
p[-1] = 0
p[0] = 0.9
rotation_interval = 100
env = VariableGamblingEnv(p=p, rotation_interval=rotation_interval)
state, _ = env.reset()

network = nn.Sequential(
    nn.Linear(len(state), env.action_space.n)
)

best_perfect_info = optimize_params(environment=env, agent_class=PerfectInfoAgent,
                                    params={'n': [len(p)], 'c': [0.25, 0.5, 1, 2, 4], 'rotation_interval': [rotation_interval]},
                                    param_search_steps=10000
                                    )
with open('data/best_perfect_info_bandit.pkl', 'wb') as f:
    pickle.dump(best_perfect_info, f)

# best_dqn = optimize_params(environment=env, agent_class=DQN,
#                            params={
#                                'network': [deepcopy(network)],
#                                'input_shape': [state.shape],
#                                'batch_size':[batch_size],
#                                'replay_buffer_size': [batch_size],
#                                'update_frequency': [1],
#                                'gamma': [0],
#                                'epsilon': [0.1],
#                                'lr': [1, 1e-1, 1e-2, 1e-3]
#                            },
#                            param_search_steps=10000
#                            )
# with open('best_dqn_bandit.pkl', 'wb') as f:
#     pickle.dump(best_dqn, f)

# best_modulated = optimize_params(environment=env, agent_class=ModulatedDQN,
#                            params={
#                                'network': [deepcopy(network)],
#                                'input_shape': [state.shape],
#                                'batch_size': [batch_size],
#                                'replay_buffer_size': [batch_size],
#                                'update_frequency': [1],
#                                'gamma': [0],
#                                'epsilon': [0.1],
#                                'lr': [1, 1e-1, 1e-2, 1e-3],
#                                'softmax_temp': [2, 1, 0.5],
#                            },
#                            param_search_steps=10000
#                            )
# with open('best_modulated_bandit.pkl', 'wb') as f:
#     pickle.dump(best_modulated, f)

best_modulated_tabular = optimize_params(environment=env, agent_class=ModulatedTabularLearner,
                                 params={'action_list': [np.arange(len(p))], 'gamma': [0.75], 'alpha': [1, 0.5, 0.1], 'default_value': [1], 'softmax_temperature': [0.25, 1]},
                                 param_search_steps=10_000
                                 )
with open('data/best_modulated_tabular_bandit.pkl', 'wb') as f:
    pickle.dump(best_modulated_tabular, f)

best_q_learner = optimize_params(environment=env, agent_class=QLearner,
                                 params={'action_list': [np.arange(len(p))], 'gamma': [0.75], 'alpha': [1, 0.5, 0.1], 'default_value': [1], 'softmax_temperature': [0.25, 1]},
                                 param_search_steps=10_000
                                 )
with open('data/best_q_learner_bandit.pkl', 'wb') as f:
    pickle.dump(best_q_learner, f)
