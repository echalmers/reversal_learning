import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import scale_luminosity


results = pd.read_csv('data/results_bandit.csv', index_col=None)
perfect_info_reward = results[results['agent'] == 'PerfectInfoAgent']['reward'].mean()
q_learning_reward = results[results['agent'] == 'QLearner']['reward'].mean()

sweep_results = pd.read_csv('data/results_bandit_sweep.csv', index_col=None)

fig = plt.figure(figsize=(6, 3))
p = sns.lineplot(sweep_results, x='tau', y='reward', n_boot=10, errorbar=('ci', 95))
p.set(xscale='log')
plt.plot([sweep_results['tau'].min(), sweep_results['tau'].max()], [perfect_info_reward] * 2, '--', c=scale_luminosity('peachpuff', 0.5))
plt.plot([sweep_results['tau'].min(), sweep_results['tau'].max()], [q_learning_reward] * 2, '--', c=scale_luminosity('palegoldenrod', 0.5))
plt.text(x=0.1, y=perfect_info_reward, s='theoretical limit', verticalalignment='top', horizontalalignment='left', c=scale_luminosity('peachpuff', 0.5))
plt.text(x=0.1, y=q_learning_reward, s='TD learning', verticalalignment='bottom', horizontalalignment='left', c=scale_luminosity('palegoldenrod', 0.5))
plt.ylabel(
"""
       reward per step
impaired ⟵        ⟶ healthy
""".strip()
)
plt.xlabel('temperature parameter (τ)')
plt.title('reward in gambling task varies with τ')

plt.tight_layout()
plt.savefig('data/tau_sweep.png', dpi=300)
plt.show()
