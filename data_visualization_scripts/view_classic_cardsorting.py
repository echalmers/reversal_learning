import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


df: pd.DataFrame = pd.read_csv('../results_classic_cardsorting.csv', index_col=None)
df['error'] = df['reward'] < 0
# calculate perseverative errors here
df = df.groupby(['algorithm', 'rep'])['error'].sum().reset_index()
df = df.append(pd.DataFrame({'algorithm': 'human', 'rep': np.arange(1000), 'error': np.random.normal(16.76, 16.71, 1000)}))
print(df)

sns.barplot(data=df, x='algorithm', y='error', errorbar='sd')
plt.show()