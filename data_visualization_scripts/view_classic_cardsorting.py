import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
pd.options.display.width = 0

np.random.seed(42)


df: pd.DataFrame = pd.read_csv('../data/results_classic_cardsorting.csv', index_col=None)
df = df.rename({'algorithm': 'agent'}, axis=1)
df['agent'] = df['agent'].replace({'SteinkeParallelLearner': 'parallel\nmodel-based/\nmodel-free',
                                   'StrategyCardSorter': 'new\nrule',
                                   'TabularCardSorter': 'TD\nlearning'})

def infer_strategy(row):
    card = np.array(eval(row['card'].replace(' ', ',')))
    strategy = np.where(card == row['action'])[0]
    return np.nan if len(strategy) == 0 else strategy[0]


df['error'] = df['reward'] < 0
df['last_strategy'] = (df['correct_strategy'] - 1) % 3
df['inferred_strategy'] = df.apply(infer_strategy, axis=1)
df['perseverative_error'] = df['inferred_strategy'] == df['last_strategy']

df = df.groupby(['agent', 'rep'])['error', 'perseverative_error'].sum().reset_index()
df = df.append(pd.DataFrame({'agent': 'human', 'rep': np.arange(1000),
                             'error': np.random.normal(16.76, 16.71, 1000),
                             'perseverative_error': np.random.normal(11.19, 11.10, 1000)}
                            ))

print(df.groupby(['agent'])['error', 'perseverative_error'].aggregate(['mean', np.std]))



layout = '''
ab
'''
fig, ax = plt.subplot_mosaic(layout, figsize=(10, 4))

plt.sca(ax['a'])
sns.barplot(data=df, x='agent', y='error', errorbar='sd',
            order=['TD\nlearning',
                   'parallel\nmodel-based/\nmodel-free',
                   'new\nrule',
                   'human'],
            palette=['palegoldenrod', 'mistyrose', 'skyblue', 'lightgrey'], edgecolor=".5"
            )
plt.ylabel('total errors')
plt.xlabel('')

plt.sca(ax['b'])
sns.barplot(data=df, x='agent', y='perseverative_error', errorbar='sd',
            order=['TD\nlearning',
                   'parallel\nmodel-based/\nmodel-free',
                   'new\nrule',
                   'human'],
            palette=['palegoldenrod', 'mistyrose', 'skyblue', 'lightgrey'], edgecolor=".5"
            )
plt.ylabel('perseverative errors')
plt.ylim(ax['a'].get_ylim())
plt.xlabel('')


plt.suptitle('performance in card sorting task for models and humans')


for label, ax in ax.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', weight='bold', va='bottom', fontfamily='serif')


plt.tight_layout()
plt.savefig('../data/classic_cardsorting_results.png', dpi=300)
plt.show()