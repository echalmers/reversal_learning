from envs.card_sorting import CardSortingEnv
from envs.multi_armed_bandit import VariableGamblingEnv
from agents.comparators import StrategyCardSorter
from agents.tabular import TabularLearner
from experiment import Experiment
import pandas as pd
pd.options.display.width = 0
import matplotlib.pyplot as plt


# env = CardSortingEnv(n_classes=4, n_features=3, change_period=10)
env = VariableGamblingEnv(p=[0.8, 0.2, 0.2], rotation_interval=10)

agent = StrategyCardSorter(action_list=[0,1,2], default_value=1, alpha=1, gamma=0, epsilon=None,
                           softmax_temperature=1, modulated_td_error=True, softmax_temperature_value_update=None)

e = Experiment(agent=TabularLearner, environment=env, n_steps=128, sensors=[lambda e, a: print(a.Q)],
               params={'action_list': [[0,1,2]], 'gamma': [0], 'alpha': [1, 0.5, 0.1], 'default_value': [1], 'softmax_temperature': [0.25, 1], 'modulated_td_error': [True]})
df = e()

print(df)
print((df['reward'] == -1).sum())
plt.plot(df['cumulative_reward'])
plt.show()