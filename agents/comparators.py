from agents.dqn import DQN
from copy import deepcopy
from envs.wrappers import OneHotEncoder
import numpy as np
import random
from agents.tabular import TabularLearner, softmax


class MBMFParallelOptimal:

    def __init__(self, mb_network, mf_network, input_shape, batch_size, replay_buffer_size, update_frequency,
                 gamma, epsilon, lr):
        self.epsilon = epsilon

        self.mb_agent = DQN(
            network=mb_network,
            input_shape=np.array([1]).shape,
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            update_frequency=1,
            gamma=gamma,
            epsilon=epsilon,
            lr=1e-1,
        )
        self.one_hot_encoder = OneHotEncoder(n_values=mf_network[-1].out_features)

        self.mf_agent = DQN(
            network=mf_network,
            input_shape=input_shape,
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            update_frequency=update_frequency,
            gamma=gamma,
            epsilon=epsilon,
            lr=lr
        )

    def select_action(self, state):
        # get mb feature selection & tranlate to action values
        feature_values = self.mb_agent.get_action_values(np.array([1])).numpy()
        one_hot_matrix, feature_vector = self.one_hot_encoder.reverse(state)
        mb_action_values = np.matmul(feature_values, one_hot_matrix)
        mb_action_values[one_hot_matrix.max(axis=0) == 0] = -float('inf')

        # get mf action values
        mf_action_values = self.mf_agent.get_action_values(state).numpy()

        # combine
        action_values = mb_action_values + mf_action_values

        # select action
        if random.random() < self.epsilon:
            return np.random.choice(len(action_values))
        return action_values.argmax()

    def update(self, state, action, reward, new_state, done):

        # infer a feature from the action & update MB learner
        _, feature_vector = self.one_hot_encoder.reverse(state)
        feature_mask = (feature_vector == action).astype(int)
        if sum(feature_mask) == 1:
            feature = random.choices(np.arange(len(feature_vector)), weights=feature_mask)[0]
            self.mb_agent.update(state=np.array([1]), action=feature, new_state=np.array([1]), reward=reward, done=done)

        # update mf learner
        self.mf_agent.update(state, action, reward, new_state, done)


class MBMFParallel2:

    def __init__(self, network, input_shape, batch_size, replay_buffer_size, update_frequency,
                 gamma, epsilon, lr, n_classes, n_features):
        self.dqn = DQN(
            network=network,
            input_shape=input_shape,
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            update_frequency=update_frequency,
            gamma=gamma,
            epsilon=epsilon,
            lr=lr
        )
        self.n_classes = n_classes
        self.n_features = n_features
        self.last_selection = None
        self.one_hot_encoder = OneHotEncoder(n_values=n_classes)

    def select_action(self, state):
        if random.random() < self.dqn.epsilon:
            self.last_selection = 'random'
            return np.random.choice(self.n_classes)

        action_values = self.dqn.get_action_values(state).numpy()
        action = action_values.argmax()
        if action >= self.n_features: # these values are for taking explicit actions, the others are for matching based on feature
            self.last_selection = 'action'
            return action - self.n_features

        self.last_selection = int(action)
        _, feature_vector = self.one_hot_encoder.reverse(state)
        return feature_vector[action]

    def update(self, state, action, reward, new_state, done):
        if self.last_selection in ['action', 'random']:
            action += self.n_features

        elif isinstance(self.last_selection, int):
            action = self.last_selection

        else:
            raise Exception()

        self.dqn.update(state, action, reward, new_state, done)


class TabularCardSorter(TabularLearner):
    def __init__(self, default_value=1, alpha=0.1, gamma=0.9, epsilon=None, softmax_temperature=1,
                 modulated_td_error=False, softmax_temperature_value_update=None):
        super().__init__(action_list=[0, 1, 2, 3], default_value=default_value, alpha=alpha, gamma=gamma, epsilon=epsilon,
                         softmax_temperature=softmax_temperature, modulated_td_error=modulated_td_error,
                         softmax_temperature_value_update=softmax_temperature_value_update)

    def select_action(self, state):
        return super().select_action((0,))

    def update(self, state, action, reward, new_state, done=None):
        return super().update((0,), action, reward, (0,), done)


class StrategyCardSorter(TabularLearner):

    def __init__(self, default_value=1, alpha=0.1, gamma=0.9, epsilon=None, softmax_temperature=1, modulated_td_error=False, softmax_temperature_value_update=None):
        super().__init__(action_list=[0, 1, 2], default_value=default_value, alpha=alpha, gamma=gamma, epsilon=epsilon,
                         softmax_temperature=softmax_temperature, modulated_td_error=modulated_td_error,
                         softmax_temperature_value_update=softmax_temperature_value_update)
        # self.last_strategy_choice = None

    def select_action(self, state):
        action = super().select_action((0,))  # select a strategy #
        # self.last_strategy_choice = action
        return state[action]  # convert strategy # into a card choice

    def get_values(self, state):
        strategy_values = [self.Q[a].get((0,), self.default_value) for a in self.Q]
        choice_values = [0, 0, 0, 0]
        for i in range(len(strategy_values)):
            strategy_points_to = state[i]
            choice_values[strategy_points_to] += strategy_values[i]
        return choice_values

    def update(self, state, action, reward, new_state, done=None):
        strategy_choice = np.where(state == action)[0]
        if len(strategy_choice) == 1:
            super().update((0,), strategy_choice[0], reward, (0,), done)
        else:
            print("action doesn't correspond to a strategy")


class SteinkeParallelLearner:

    def __init__(self, default_value=1, alpha_mb=0.1, inertia_mb=0.9, softmax_temperature=1,
                 alpha_mf=0.1, inertia_mf=0.9, w=0.5):
        self.inertia_mb = inertia_mb
        self.inertia_mf = inertia_mf
        self.w = w
        self.softmax_temperature = softmax_temperature

        self.mb_learner = StrategyCardSorter(default_value=default_value, alpha=alpha_mb, gamma=0, softmax_temperature=softmax_temperature,
                                             modulated_td_error=False, softmax_temperature_value_update=None)
        self.mf_learner = TabularCardSorter(default_value=default_value, alpha=alpha_mf, gamma=0, softmax_temperature=softmax_temperature,
                                         modulated_td_error=False, softmax_temperature_value_update=None)

    def select_action(self, state):
        mb_values = np.array(self.mb_learner.get_values(state))
        mf_values = np.array(self.mf_learner.get_values(state))
        combined_values = self.w * mb_values + (1 - self.w) * mf_values
        action = np.random.choice([0, 1, 2, 3], p=softmax(combined_values, self.softmax_temperature))
        if action not in state:
            print('picking an action with no strategy')
        return action

    def update(self, state, action, reward, new_state, done=None):
        for table, inertia in [(self.mb_learner.Q, self.inertia_mb), (self.mf_learner.Q, self.inertia_mf)]:
            for d in table.values():
                for s in d:
                    d[s] *= inertia

        self.mf_learner.update(state, action, reward, new_state, done)
        self.mb_learner.update(state, action, reward, new_state, done)
