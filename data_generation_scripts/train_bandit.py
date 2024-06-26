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
import pickle


n = 7
p = np.random.rand(n) * 0.5 + 0.25
p[-1] = 0
p[0] = 0.9
env = VariableGamblingEnv(p=p, rotation_interval=100)
state, _ = env.reset()


# with open('best_dqn_bandit.pkl', 'rb') as f:
#     standard_agent = pickle.load(f)
# with open('best_modulated_bandit.pkl', 'rb') as f:
#     modulated_agent = pickle.load(f)
with open('../data/best_perfect_info_bandit.pkl', 'rb') as f:
    perfect_info = pickle.load(f)
with open('../data/best_modulated_tabular_bandit.pkl', 'rb') as f:
    modulated_tabular = pickle.load(f)
with open('../data/best_q_learner_bandit.pkl', 'rb') as f:
    q_learner = pickle.load(f)
with open('../data/best_gradient_agent.pkl', 'rb') as f:
    gradient_agent = pickle.load(f)



agents = [
    #modulated_agent,
    # standard_agent,
    perfect_info,
    modulated_tabular,
    q_learner,
    gradient_agent
]

results = []

for rep in range(10):
    for agent in agents:
        # np.random.seed(42)
        # random.seed(42)

        STEPS = 5000
        state, _ = env.reset()
        reward_history = np.zeros(STEPS)
        for step in range(STEPS):

            correct_action = env.get_correct_action()  # just for results - agent can't see it
            action = agent.select_action(state)
            new_state, reward, done, _, _ = env.step(action)
            results.append({'rep': rep, 'agent': agent.__class__.__name__, 'step': step, 'correct_action': correct_action, 'action': action, 'reward': reward})

            reward_history[step] = reward

            agent.update(state=state, action=action, reward=reward, new_state=new_state, done=done)

            state = new_state

        plt.plot(reward_history.cumsum())

pd.DataFrame(results).to_csv('../data/results_bandit.csv', index=False)

plt.legend([x.__class__.__name__ for x in agents])
plt.show()
