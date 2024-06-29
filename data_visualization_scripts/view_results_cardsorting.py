import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import scale_luminosity
import matplotlib.transforms as mtransforms
from scipy.stats import ttest_ind
from itertools import combinations

results = pd.read_csv('../data/results_cardsorting.csv', index_col=None)
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
sns.lineplot(data=results, x='step', y='cumulative_reward', hue='agent', n_boot=10, errorbar=('ci', 95),
             palette=[scale_luminosity('skyblue', 0.5), scale_luminosity('palegoldenrod', 0.5), scale_luminosity('mistyrose', 0.75)])
plt.ylabel('cumulative reward')
# plt.title('cumulative reward in card sorting task')

plt.sca(ax['b'])
results['agent'] = results['agent'].apply(lambda s: s.replace(' ', '\n').replace('/', '/\n'))
sns.barplot(data=results, x='agent', y='reward', edgecolor=".5", palette=['skyblue', 'palegoldenrod', 'mistyrose'], errorbar=('ci', 95))
plt.xlabel('')
plt.ylabel('average reward per step')
# plt.title('average reward per step')

plt.suptitle('model performance in card sorting task')

for label, ax in ax.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', weight='bold', va='bottom', fontfamily='serif')

# compute statistical significance
mean_reward_ss = pd.DataFrame()
mean_rewards_per_step = results.groupby(['agent', 'rep'])['reward'].mean().reset_index()
print(mean_rewards_per_step.groupby('agent')['reward'].aggregate(['mean', 'std']).round(2))
for comparison in combinations(results['agent'].unique(), 2):
    print('____', comparison, '____')
    rewards_0 = mean_rewards_per_step[mean_rewards_per_step['agent'] == comparison[0]]['reward']
    rewards_1 = mean_rewards_per_step[mean_rewards_per_step['agent'] == comparison[1]]['reward']
    # print('compare mean rewards', ttest_ind(rewards_0, rewards_1, equal_var=False))
    mean_reward_test = ttest_ind(rewards_0, rewards_1, equal_var=False)
    mean_reward_ss.loc[comparison[0], comparison[1]] = f'{mean_reward_test.statistic:.2f}, p={mean_reward_test.pvalue:.2E}, df={mean_reward_test.df:.0f}'
print(mean_reward_ss)

plt.tight_layout()
plt.savefig('../data/cardsorting_results.png', dpi=300)
plt.show()
