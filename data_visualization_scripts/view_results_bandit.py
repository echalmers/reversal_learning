import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import scale_luminosity
import matplotlib.transforms as mtransforms
from scipy.stats import ttest_ind
from itertools import combinations


# compute win-stay and lose-switch, and perseveration errors FOR BANDIT TASK ONLY
results = pd.read_csv('../data/results_bandit.csv', index_col=None)
results['agent'] = results['agent'].replace({'PerfectInfoAgent': 'theoretical limit',
                                             'NewLearner': 'new rule',
                                             'QLearner': 'TD learning'})

# infer rotation inteval
diff = results['correct_action'].diff()
interval = diff[diff != 0].index[1]

# calculate LSWS
results['switch'] = results['action'].diff() != 0
results['loose_switch'] = ((results['reward'] < 0) & (results['switch'])) * 100
results['win_stay'] = ((results['reward'] > 0) & (~results['switch'])) * 100
wsls = pd.DataFrame()
wsls['loose_switch'] = results[results['reward'] < 0].groupby(['agent', 'rep'])['loose_switch'].mean()
wsls['win_stay'] = results[results['reward'] > 0].groupby(['agent', 'rep'])['win_stay'].mean()
wsls = wsls.reset_index()
print(wsls)

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
sns.lineplot(data=results[results['step'] < 1000], x='step', y='cumulative_reward', hue='agent', n_boot=10, errorbar=('ci', 95),
             palette=[scale_luminosity('peachpuff', 0.5), scale_luminosity('skyblue', 0.5), scale_luminosity('palegoldenrod', 0.5)])
plt.ylabel('cumulative reward')
plt.title('cumulative reward in variable Iowa gambling task')

results['agent'] = results['agent'].apply(lambda s: s.replace(' ', '\n'))
wsls['agent'] = wsls['agent'].apply(lambda s: s.replace(' ', '\n'))

plt.sca(ax['b'])
sns.barplot(results, x='agent', y='reward', edgecolor=".5", palette=['peachpuff', 'skyblue', 'palegoldenrod'], errorbar=('ci', 95),)
plt.title('reward obtained')
plt.ylabel('average reward per step')
plt.xlabel('')

plt.sca(ax['c'])
sns.barplot(wsls, x='agent', y='loose_switch', edgecolor=".5", palette=['peachpuff', 'skyblue', 'palegoldenrod'], errorbar=('ci', 95),
            order=['theoretical\nlimit','new\nrule', 'TD\nlearning'])
plt.title('lose-switch')
plt.ylabel('lose-switch (%)')
plt.xlabel('')
plt.ylim((0, 100))
# plt.xticks(rotation=25, horizontalalignment='right')

plt.sca(ax['d'])
sns.barplot(wsls, x='agent', y='win_stay', edgecolor=".5", palette=['peachpuff', 'skyblue', 'palegoldenrod'], errorbar=('ci', 95),
            order=['theoretical\nlimit','new\nrule', 'TD\nlearning'])
plt.title('win-stay')
plt.ylabel('win-stay (%)')
plt.xlabel('')
plt.ylim((0, 100))

for label, ax in ax.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', weight='bold', va='bottom', fontfamily='serif')

# calculate statistical significance
mean_rewards_per_step = results.groupby(['agent', 'rep'])['reward'].mean().reset_index()

print(wsls.groupby('agent')['win_stay', 'loose_switch'].aggregate(['mean', 'std']))
print(mean_rewards_per_step.groupby('agent')['reward'].aggregate(['mean', 'std']))

for comparison in combinations(results['agent'].unique(), 2):
    print('____', comparison, '____')
    rewards_0 = mean_rewards_per_step[mean_rewards_per_step['agent'] == comparison[0]]['reward']
    rewards_1 = mean_rewards_per_step[mean_rewards_per_step['agent'] == comparison[1]]['reward']
    print('compare mean rewards', ttest_ind(rewards_0, rewards_1, equal_var=False))
    print('compare lose-switch', ttest_ind(wsls[wsls['agent'] == comparison[0]]['loose_switch'], wsls[wsls['agent'] == comparison[1]]['loose_switch'], equal_var=False))
    print('compare win-stay', ttest_ind(wsls[wsls['agent'] == comparison[0]]['win_stay'], wsls[wsls['agent'] == comparison[1]]['win_stay'], equal_var=False))



plt.tight_layout()
plt.savefig('../data/bandit_results.png', dpi=300)
plt.show()

