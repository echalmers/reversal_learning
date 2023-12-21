import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import scale_luminosity

results = pd.read_csv('data/results_cardsorting.csv', index_col=None)
results['agent'] = results['agent'].replace({'DQN': 'TD learning network',
                                             'ModulatedDQN': 'new rule',
                                             'MBMFParallel2': 'parallel model-based/model-free'})

results['cumulative_reward'] = results.groupby(['rep', 'agent'])['reward'].transform(pd.Series.cumsum)

layout = '''
aab
'''
fig, ax = plt.subplot_mosaic(layout, figsize=(10, 4))

print(results)
plt.sca(ax['a'])
sns.lineplot(data=results, x='step', y='cumulative_reward', hue='agent', n_boot=10,
             palette=[scale_luminosity('skyblue', 0.5), scale_luminosity('palegoldenrod', 0.5), scale_luminosity('mistyrose', 0.75)])
plt.ylabel('')
plt.title('cumulative reward in card sorting task')

plt.sca(ax['b'])
results['agent'] = results['agent'].apply(lambda s: s.replace(' ', '\n').replace('/', '/\n'))
sns.barplot(data=results, x='agent', y='reward', edgecolor=".5", palette=['skyblue', 'palegoldenrod', 'mistyrose'])
plt.xlabel('')
plt.ylabel('')
plt.title('average reward per step')

plt.tight_layout()
plt.savefig('data/cardsorting_results.png', dpi=300)
plt.show()
