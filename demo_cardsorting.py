from envs.card_sorting import CardSortingEnv
from envs.wrappers import OneHotWrapper
from agents.tabular import TabularLearner
from agents.dqn import DQN
from experiment import Experiment
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn


env = CardSortingEnv(n_classes=4, n_features=3, change_period=10)
env = OneHotWrapper(env, env.n_classes)

state, _ = env.reset()

for _ in range(10):
    state, reward, terminated, truncated, info = env.step(0)
    print(state, env.match_feature, env.card[env.match_feature])
exit()


network = nn.Sequential(
    nn.Linear(len(state), env.action_space.n)
)
agent = DQN(network=network,
            input_shape=state.shape,
            batch_size=10,
            replay_buffer_size=4,
            update_frequency=1,
            gamma=0,
            epsilon=0,
            lr=0.01,
            softmax_temp=10  # new rule
)

e = Experiment(agent=agent, environment=env, n_steps=500)
df = e()

plt.plot(df['cumulative_reward'])
plt.show()
