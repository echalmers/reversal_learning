from envs.card_sorting import CardSortingEnv
from envs.multi_armed_bandit import VariableGamblingEnv
from agents.comparators import StrategyCardSorter, SteinkeParallelLearner, TabularCardSorter
from agents.tabular import TabularLearner, QLearner, NewLearner
from experiment import Experiment
import pandas as pd
pd.options.display.width = 0
import matplotlib.pyplot as plt


# import random
# random.seed(42)
# import numpy as np
# np.random.seed(42)




if __name__ == '__main__':

    results = pd.DataFrame()
    for rep in range(20):

        env = VariableGamblingEnv(p=[0.9] * 4, reward=[1, 1, 0.5, 0.5], penalty=[-12.5, -12.5, -2.5, -2.5], rotation_interval=50)

        for experiment in [
            Experiment(agent=NewLearner, environment=env, n_steps=1000,
                       params={'action_list': [[0,1,2,3]], 'default_value': [1], 'gamma': [0], 'alpha': [1, 0.5, 0.1], 'epsilon': [None],
                               'softmax_temperature': [0.25, 1]}),

            Experiment(agent=QLearner, environment=env, n_steps=1000,
                       params={'action_list': [[0, 1, 2, 3]], 'default_value': [1], 'gamma': [0], 'alpha': [1, 0.5, 0.1], 'epsilon': [None],
                               'softmax_temperature': [0.25, 1]}),
        ]:
            df = experiment.run()
            df['rep'] = rep
            results = pd.concat((results, df))

            results.to_csv('../results_classic_IGT.csv', index=False)
    print(results)