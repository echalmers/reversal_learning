import time
import itertools
from copy import deepcopy
import numpy as np
import pandas as pd


class ParamEvaluator:

    def __init__(self, environment, agent_class, param_search_steps=1000):
        self.environment = deepcopy(environment)
        self.agent_class = agent_class
        self.param_search_steps = param_search_steps

    def __call__(self, params):
        experiment = Experiment(self.environment, self.agent_class(**params), n_steps=self.param_search_steps)
        results = experiment()
        return results['reward'].sum()


def optimize_params(environment, agent_class, params, param_search_steps=1000):
    combinations = [dict(zip(list(params), combination)) for combination in itertools.product(*params.values())]
    if len(combinations) == 1:
        return agent_class(**combinations[0])
    else:
        rewards = list(map(ParamEvaluator(environment, agent_class, param_search_steps), combinations))
        best_idx = np.argmax(rewards)
        print(combinations[best_idx])
        return agent_class(**combinations[best_idx])


class Experiment:

    def __init__(self, environment, agent, n_steps, params=None, param_search_steps=1000, sensors=tuple()):

        self.environment = deepcopy(environment)
        self.agent = deepcopy(agent)
        self.sensors = sensors

        self.params = params
        self.param_search_steps = param_search_steps

        self.state, _ = environment.reset()
        self.n_steps = n_steps

    def run(self):
        return self()

    def __call__(self):

        if self.params is not None:
            self.agent = optimize_params(self.environment, self.agent, self.params, self.param_search_steps)

        # setup results dataframe
        results = pd.DataFrame(index=range(self.n_steps))
        results['experiment_id'] = time.time()
        results['algorithm'] = self.agent.__class__.__name__
        results['step'] = np.arange(self.n_steps)
        results['reward'] = 0
        results['action'] = 0
        if hasattr(self.environment, 'p'):
            results['n'] = len(self.environment.p)
        if hasattr(self.environment, 'n_classes'):
            results['n_classes'] = self.environment.n_classes
        if hasattr(self.environment, 'n_classes'):
            results['n_features'] = self.environment.n_features

        for sensor in self.sensors:
            results[sensor.__name__] = None

        for step in range(self.n_steps):

            action = self.agent.select_action(self.state)
            new_state, reward, done, _, _ = self.environment.step(action)

            self.agent.update(self.state, action, reward, new_state, done)
            results.loc[step, 'reward'] = reward
            results.loc[step, 'action'] = action

            self.state = new_state
            for sensor in self.sensors:
                results._set_value(step, sensor.__name__, sensor(self.environment, self.agent))

        results['cumulative_reward'] = results['reward'].cumsum()
        return results
