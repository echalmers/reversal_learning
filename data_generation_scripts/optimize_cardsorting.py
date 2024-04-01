from experiment import optimize_params
from envs.wrappers import OneHotWrapper
from envs.card_sorting import CardSortingEnv
from torch import nn
from agents.dqn import DQN
from agents.comparators import MBMFParallel2
from agents.modulated_dqn import ModulatedDQN
from copy import deepcopy
import pickle


env = CardSortingEnv(n_classes=4, n_features=3, change_period=100)
env = OneHotWrapper(env, env.n_classes)

state, _ = env.reset()

network = nn.Sequential(
    nn.Linear(len(state), env.action_space.n)
)


best_dqn = optimize_params(environment=env, agent_class=DQN,
                           params={
                               'network': [deepcopy(network)],
                               'input_shape': [state.shape],
                               'batch_size':[10],
                               'replay_buffer_size': [10],
                               'update_frequency': [1],
                               'gamma': [0],
                               'epsilon': [0.1],
                               'lr': [1, 1e-1, 1e-2, 1e-3]
                           },
                           param_search_steps=10000
                           )
with open('../data/best_dqn_cardsorting.pkl', 'wb') as f:
    pickle.dump(best_dqn, f)

best_modulated = optimize_params(environment=env, agent_class=ModulatedDQN,
                           params={
                               'network': [deepcopy(network)],
                               'input_shape': [state.shape],
                               'batch_size':[10],
                               'replay_buffer_size': [10],
                               'update_frequency': [1],
                               'gamma': [0],
                               'epsilon': [0.1],
                               'lr': [1, 1e-1, 1e-2, 1e-3],
                               'softmax_temp': [2, 1, 0.5],
                           },
                           param_search_steps=10000
                           )
with open('../data/best_modulated_cardsorting.pkl', 'wb') as f:
    pickle.dump(best_modulated, f)

best_parallel = optimize_params(environment=env, agent_class=MBMFParallel2,
                           params={
                               'network': [nn.Sequential(nn.Linear(len(state), env.unwrapped.n_features + env.action_space.n))],
                               'input_shape': [state.shape],
                               'batch_size':[10],
                               'replay_buffer_size': [10],
                               'update_frequency': [1],
                               'gamma': [0],
                               'epsilon': [0.1],
                               'lr': [1, 1e-1, 1e-2, 1e-3],
                               'n_classes': [env.unwrapped.n_classes],
                               'n_features': [env.unwrapped.n_features]
                           },
                           param_search_steps=10000
                           )
with open('../data/best_parallel_cardsorting.pkl', 'wb') as f:
    pickle.dump(best_parallel, f)