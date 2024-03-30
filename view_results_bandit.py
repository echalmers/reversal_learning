import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import scale_luminosity
import matplotlib.transforms as mtransforms


# compute win-stay and lose-switch, and perseveration errors FOR BANDIT TASK ONLY
results = pd.read_csv('data/results_bandit.csv', index_col=None)
results['agent'] = results['agent'].replace({'PerfectInfoAgent': 'theoretical limit',
                                             'NewLearner': 'new rule',
                                             'QLearner': 'TD learning'})

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
results['loose_switch'] = ((results['reward'] < 0) & (results['switch'])) * 100
results['win_stay'] = ((results['reward'] > 0) & (~results['switch'])) * 100
wsls = pd.DataFrame()
wsls['loose_switch'] = results[results['reward'] < 0].groupby('agent')['loose_switch'].mean() * 100
wsls['win_stay'] = results[results['reward'] > 0].groupby('agent')['win_stay'].mean() * 100

# compute cumulative reward
results['cumulative_reward'] = results.groupby(['rep', 'agent'])['reward'].transform(pd.Series.cumsum)


layout = '''
aaaaa
.....
b.c.d
'''
fig, ax = plt.subplot_mosaic(layout, width_ratios=[10,0,10,0,10], height_ratios=[10,0.1,10], figsize=(9, 6))

# plt.suptitle('performance in variable Iowa Gambling Task')

plt.sca(ax['a'])
sns.lineplot(data=results[results['step'] < 1000], x='step', y='cumulative_reward', hue='agent', n_boot=1, errorbar=('ci', 95),
             palette=[scale_luminosity('peachpuff', 0.5), scale_luminosity('skyblue', 0.5), scale_luminosity('palegoldenrod', 0.5)])
plt.ylabel('cumulative reward')
plt.title('cumulative reward in variable Iowa gambling task')

results['agent'] = results['agent'].apply(lambda s: s.replace(' ', '\n'))
plt.sca(ax['b'])
sns.barplot(results, x='agent', y='reward', edgecolor=".5", palette=['peachpuff', 'skyblue', 'palegoldenrod'], errorbar=('ci', 95),)
plt.title('reward obtained')
plt.ylabel('average reward per step')
plt.xlabel('')

plt.sca(ax['c'])
sns.barplot(results[results['reward'] < 0], x='agent', y='loose_switch', edgecolor=".5", palette=['peachpuff', 'skyblue', 'palegoldenrod'], errorbar=('ci', 95),)
plt.title('lose-switch')
plt.ylabel('lose-switch (%)')
plt.xlabel('')
plt.ylim((0, 100))
# plt.xticks(rotation=25, horizontalalignment='right')

plt.sca(ax['d'])
sns.barplot(results[results['reward'] > 0], x='agent', y='win_stay', edgecolor=".5", palette=['peachpuff', 'skyblue', 'palegoldenrod'], errorbar=('ci', 95),)
plt.title('win-stay')
plt.ylabel('win-stay (%)')
plt.xlabel('')
plt.ylim((0, 100))

for label, ax in ax.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', weight='bold', va='bottom', fontfamily='serif')

plt.tight_layout()
plt.savefig('data/bandit_results.png', dpi=300)
plt.show()

