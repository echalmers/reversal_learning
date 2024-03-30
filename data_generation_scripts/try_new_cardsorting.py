from envs.card_sorting import CardSortingEnv
from envs.multi_armed_bandit import VariableGamblingEnv
from agents.comparators import StrategyCardSorter, SteinkeParallelLearner, TabularCardSorter
from agents.tabular import TabularLearner
from experiment import Experiment
import pandas as pd
pd.options.display.width = 0
import matplotlib.pyplot as plt


# import random
# random.seed(42)
# import numpy as np
# np.random.seed(42)


def correct_strategy(env, agent):
    return env.match_feature

def card(env, agent):
    return env.card


if __name__ == '__main__':

    results = pd.DataFrame()
    for rep in range(20):

        env = CardSortingEnv(n_classes=4, n_features=3, change_period=10)

        for experiment in [
            Experiment(agent=TabularCardSorter, environment=env, n_steps=128, sensors=[correct_strategy, card],
                       params={'gamma': [0], 'alpha': [1, 0.5, 0.1], 'default_value': [1],
                               'softmax_temperature': [0.25, 1], 'modulated_td_error': [False]}),

            Experiment(agent=StrategyCardSorter, environment=env, n_steps=128, sensors=[correct_strategy, card],
                       params={'gamma': [0], 'alpha': [1, 0.5, 0.1], 'default_value': [1],
                               'softmax_temperature': [0.25, 1], 'modulated_td_error': [True]}),

            Experiment(agent=SteinkeParallelLearner, environment=env, n_steps=128, sensors=[correct_strategy, card],
                       params={'default_value': [1], 'alpha_mb': [1, 0.5, 0.1], 'inertia_mb': [0.9, 0.5, 0.1],
                               'softmax_temperature': [0.25, 1],
                               'alpha_mf': [1, 0.5, 0.1], 'inertia_mf': [0.9, 0.5, 0.1], 'w': [0.9, 0.5, 0.1]})
        ]:
            df = experiment.run()
            df['rep'] = rep
            results = pd.concat((results, df))

            results.to_csv('../results_classic_cardsorting.csv', index=False)