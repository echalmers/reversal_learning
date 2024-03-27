from agents.tabular import TabularLearner
from envs.multi_armed_bandit import VariableGamblingEnv
from experiment import Experiment
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.width = 0
import seaborn as sns

agent = TabularLearner([0,1], default_value=1, alpha=0.2, gamma=0.0, epsilon=None,
                       softmax_temperature=1, modulated_td_error=True)

def get_value_of_zero(env, agent):
    return agent.Q[0].get((0,), agent.default_value)

results = pd.DataFrame()

for p in [
    (0.8, 0.2),
    (0.2, 0.8),
    (0.7, 0.3),
    (0.3, 0.7),
    (0.6, 0.4),
    (0.4, 0.6)
]:
    e = Experiment(environment=VariableGamblingEnv(p=p, reward=1, penalty=0), agent=agent, n_steps=100, sensors=[get_value_of_zero])
    df = e()
    agent = e.agent

    results = pd.concat((results, df), ignore_index=True)

results['pref'] = results['action'].rolling(10).mean()
print(results)
plt.plot(results['get_value_of_zero'])
# plt.plot(results['pref'])
plt.show()