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

# infer rotation inteval
diff = results['correct_action'].diff()
interval = diff[diff != 0].index[1]

# calculate LSWS
results['switch'] = results['action'].diff() != 0
results['loose_switch'] = (results['reward'] < 0) & (results['switch'])
results['win_stay'] = (results['reward'] > 0) & (~results['switch'])
print(results[results['reward'] < 0].groupby('agent')['loose_switch'].mean())
print(results[results['reward'] > 0].groupby('agent')['win_stay'].mean())

results['cumulative_reward'] = results.groupby(['rep', 'agent'])['reward'].transform(pd.Series.cumsum)
sns.lineplot(data=results, x='step', y='cumulative_reward', hue='agent', n_boot=1)
plt.show()

