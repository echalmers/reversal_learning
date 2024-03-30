import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


df: pd.DataFrame = pd.read_csv('../results_classic_IGT.csv', index_col=None)
sns.lineplot(data=df, x='step', y='cumulative_reward', hue='algorithm', n_boot=1)
plt.show()


for deck in [0, 1, 2, 3]:
    df[deck] = (df['action'] == deck).astype(int)

df = df.groupby(['algorithm', 'rep'])[0, 1, 2, 3, 'reward'].sum().reset_index()
df['score'] = df[2] + df[3] - df[0] - df[1]



print(df)

sns.barplot(data=df, x='algorithm', y='reward', errorbar='sd')
plt.show()