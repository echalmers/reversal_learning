import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results = pd.read_csv('results_cardsorting.csv', index_col=None)
results['cumulative_reward'] = results.groupby(['rep', 'agent'])['reward'].transform(pd.Series.cumsum)

print(results)
sns.lineplot(data=results, x='step', y='cumulative_reward', hue='agent', n_boot=10)
plt.show()
