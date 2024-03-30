from numbers import Number

import numpy as np


def softmax(x: np.array, t):
    x_local = np.array(x).astype(float)
    if x_local.ndim == 1:
        return np.exp(x_local/t) / np.exp(x_local/t).sum()
    else:
        if not isinstance(t, Number):
            t = np.array(t)[:, None]
        return np.exp(x_local/t) / np.exp(x_local/t).sum(axis=1).reshape(-1, 1)


class TabularLearner:  #(ReinforcementLearner):
    """ basic tabular reinforcement learner. Optionally implements the new rule from our paper:
    Brain-Inspired modulation of reward-prediction error improves reinforcement learning adaptation to environmental
    change
    """

    def __init__(self, action_list, default_value=1, alpha=0.1, gamma=0.9, epsilon=None,
                 softmax_temperature=1, modulated_td_error=False, softmax_temperature_value_update=None):
        """
        :param action_list: list of available actions in the environment
        :param default_value: default value assigned to previously-untried state-action combinations
        :param alpha: learning rate
        :param gamma: discount factor
        :param epsilon: parameter for e-greedy action selection
        :param softmax_temperature: softmax temperature for use in the new RL rule. Will also be used for action
        selection if specified and epsilon is not
        :param modulated_td_error: if True, will use the new RL rule from the paper
        :param softmax_temperature_value_update: softmax temperature to use for value updates.
        Defaults to softmax_tempurature
        """
        self.Q = {action: dict() for action in action_list}
        self.alpha = alpha
        self.gamma = gamma
        self.default_value = default_value
        self.epsilon = epsilon
        self.softmax_temperature = softmax_temperature
        self.modulated_td_error = modulated_td_error
        self.softmax_temperature_value_update = softmax_temperature_value_update or self.softmax_temperature
        self.sum_update = 0

    def select_action(self, state):
        if isinstance(self.epsilon, Number):
            if np.random.random() < self.epsilon:
                return np.random.choice(list(self.Q))

        elif isinstance(self.softmax_temperature, Number):
            try:
                return np.random.choice(list(self.Q),
                                        p=softmax([self.Q[a].get(tuple(state), self.default_value) for a in self.Q], t=self.softmax_temperature)
                                        )
            except Exception as ex:
                raise ex

        return max(self.Q, key=lambda a: self.Q[a].get(tuple(state), self.default_value))

    def update(self, state, action, reward, new_state, done=None):
        current_val = self.Q[action].get(tuple(state), self.default_value)
        observed_val = reward + self.gamma * (max([self.Q[a].get(tuple(new_state), self.default_value) for a in self.Q]))

        if self.modulated_td_error:
            p_act = softmax(
                [self.Q[a].get(tuple(state), self.default_value) for a in self.Q],
                t=self.softmax_temperature_value_update
            )[list(self.Q).index(action)]

            self.Q[action][tuple(state)] = current_val + self.alpha * p_act * (observed_val - current_val)
            self.sum_update += self.alpha * p_act

        else:
            self.Q[action][tuple(state)] = current_val + self.alpha * (observed_val - current_val)
            self.sum_update += self.alpha

    def get_values(self, state):
        return [self.Q[a].get(tuple(state), self.default_value) for a in self.Q]

class QLearner(TabularLearner):

    def __init__(self, **kwargs):
        kwargs['modulated_td_error'] = False
        super().__init__(**kwargs)


class NewLearner(TabularLearner):

    def __init__(self, **kwargs):
        kwargs['modulated_td_error'] = True
        super().__init__(**kwargs)
