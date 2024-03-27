from agents.tabular import TabularLearner
from envs.multi_armed_bandit import VariableGamblingEnv
from experiment import Experiment
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


agent = TabularLearner([0,1], default_value=1, alpha=0.1, gamma=0.0, epsilon=None,
                       softmax_temperature=0.1, modulated_td_error=True)

sp1 = VariableGamblingEnv(p=[0.8, 0.8], reward=[1, 0.5], penalty=[-5.25, -0.75])  # DA
sp2 = VariableGamblingEnv(p=[0.8, 0.8], reward=[1, 0.5], penalty=[-2.75, -3.25])  # AD

lp1 = VariableGamblingEnv(p=[0.8, 0.8], reward=[1, 0.5], penalty=[-2.75, -3.25])  # AD
lp2 = VariableGamblingEnv(p=[0.8, 0.8], reward=[1, 0.5], penalty=[-5.25, -0.75])  # DA

results = pd.DataFrame()
for rep in range(100):
    print(rep)
    for task_set in [(sp1, sp2, 'sp'), (lp1, lp2, 'lp')]:
        e1 = Experiment(environment=task_set[0], agent=agent, n_steps=100)
        df1 = e1()
        df1['block'] = 1

        e2 = Experiment(environment=task_set[1], agent=e1.agent, n_steps=100)
        df2 = e2()
        df2['step'] += e1.n_steps
        df2['block'] = 2

        df = pd.concat((df1, df2), ignore_index=True)
        df['version'] = task_set[2]
        results = pd.concat((results, df), ignore_index=True)


print(results)
print(results.groupby(['version', 'block'])['action'].mean())
sns.lineplot(data=results, x='step', y='cumulative_reward', hue='version', n_boot=1)
plt.show()