import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# compute win-stay and lose-switch, and perseveration errors FOR BANDIT TASK ONLY
results = pd.read_csv('results_bandit.csv', index_col=None)
# for i in range(1, len(results.index)):
#     last_act = results.loc[i - 1, 'action']
#     this_act = results.loc[i, 'action']
#     correct_act = results.loc[i, 'correct_action']
#     last_reward = results.loc[i - 1, 'reward']
#     results.loc[i, 'ws'] = int(last_reward > 0 and this_act == last_act)
#     results.loc[i, 'ls'] = int(last_reward <= 0 and this_act != last_act)
#     results.loc[i, 'perseveration'] = int(this_act == (correct_act - 1) % results['correct_action'].max())
# print(results.groupby('agent')[['ws', 'ls', 'perseveration']].sum())

results['cumulative_reward'] = results.groupby(['rep', 'agent'])['reward'].transform(pd.Series.cumsum)
sns.lineplot(data=results, x='step', y='cumulative_reward', hue='agent', n_boot=1)
plt.show()

