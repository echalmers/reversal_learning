import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import scale_luminosity
from scipy.stats import ttest_ind
pd.options.display.width = 0


fig = plt.figure(figsize=(6, 3))

results = pd.read_csv('../data/results_cardsorting_paradox.csv', index_col=None)
sns.lineplot(data=results, x='n_features', y='reward', hue='agent', palette=[scale_luminosity('palegoldenrod', 0.5), scale_luminosity('skyblue', 0.5)], errorbar=('ci', 95))
plt.legend([],[], frameon=False)
plt.ylabel('')
plt.xlabel('number of card features')
plt.title('performance in cardsorting tasks of increasing size')
plt.ylabel('average reward per choice')
plt.text(x=4, y=0.2, horizontalalignment='left', verticalalignment='bottom', s='new rule', c=scale_luminosity('skyblue', 0.5))
plt.text(x=8, y=-0.5, horizontalalignment='left', verticalalignment='bottom', s='TD learning', c=scale_luminosity('palegoldenrod', 0.3))


# compute statistical significance
table = pd.DataFrame(columns=['task features', 'reward: TD learning', 'reward: new rule', 't-test results'])
for n_features in results['n_features'].unique():
    data_0 = results[(results['n_features'] == n_features) & (results['agent'] == 'TD learning')]['reward']
    data_1 = results[(results['n_features'] == n_features) & (results['agent'] == 'new rule')]['reward']
    ttest = ttest_ind(data_0, data_1, equal_var=False)
    # print(n_features, ttest_ind(data_0, data_1, equal_var=False))
    table.loc[n_features] = [
        n_features,
        f'{data_0.mean():.2f} ({data_0.std():.2f})',
        f'{data_1.mean():.2f} ({data_1.std():.2f})',
        f'{ttest.statistic:.2f}, p={ttest.pvalue:.2E}, df={ttest.df:.0f}'
    ]
print(table)

plt.tight_layout()
plt.savefig('../data/paradox.png', dpi=300)
plt.show()