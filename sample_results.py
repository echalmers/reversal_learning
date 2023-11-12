import pandas as pd


# compute win-stay and lose-switch, and perseveration errors FOR BANDIT TASK ONLY
df = pd.read_csv('results.csv', index_col=None)
for i in range(1, len(df.index)):
    last_act = df.loc[i-1, 'action']
    this_act = df.loc[i, 'action']
    last_reward = df.loc[i-1, 'reward']
    df.loc[i, 'ws'] = int(last_reward > 0 and this_act == last_act)
    df.loc[i, 'ls'] = int(last_reward <= 0 and this_act != last_act)
print(df.groupby('agent')[['ws', 'ls']].sum())

