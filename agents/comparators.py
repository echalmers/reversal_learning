from agents.dqn import DQN
from copy import deepcopy
from envs.wrappers import OneHotEncoder
import numpy as np
import random


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
