import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import scale_luminosity


fig = plt.figure(figsize=(6, 3))

results = pd.read_csv('data/results_cardsorting_paradox.csv', index_col=None)
sns.lineplot(data=results, x='n_features', y='reward', hue='agent', palette=[scale_luminosity('palegoldenrod', 0.5), scale_luminosity('skyblue', 0.5)], errorbar=('ci', 95))
plt.legend([],[], frameon=False)
plt.ylabel('')
plt.xlabel('number of card features')
plt.title('performance in cardsorting tasks of increasing size')
plt.ylabel('average reward per choice')
plt.text(x=4, y=0.2, horizontalalignment='left', verticalalignment='bottom', s='new rule', c=scale_luminosity('skyblue', 0.5))
plt.text(x=8, y=-0.5, horizontalalignment='left', verticalalignment='bottom', s='TD learning', c=scale_luminosity('palegoldenrod', 0.3))

plt.tight_layout()
plt.savefig('data/paradox.png', dpi=300)
plt.show()