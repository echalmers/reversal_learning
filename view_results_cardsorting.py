import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results = pd.read_csv('results_cardsorting.csv', index_col=None)
results['agent'] = results['agent'].replace({'DQN': 'TD learning network',
                                             'ModulatedDQN': 'new rule',
                                             'MBMFParallel2': 'parallel\nmodel-based/model-free'})

results['cumulative_reward'] = results.groupby(['rep', 'agent'])['reward'].transform(pd.Series.cumsum)

print(results)
# sns.lineplot(data=results, x='step', y='cumulative_reward', hue='agent', n_boot=10)
sns.barplot(data=results, x='agent', y='reward', edgecolor=".5", palette=['skyblue', 'palegoldenrod', 'mistyrose'])
plt.xlabel('')
plt.ylabel('')
plt.title('average reward per step - card sorting task')
plt.show()
