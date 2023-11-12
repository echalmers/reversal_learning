import gymnasium as gym
import numpy as np


class OneHotEncoder:

    def __init__(self, n_values):
        self.n_values = n_values

    def __call__(self, vector):
        new_obs = np.zeros((len(vector), self.n_values))
        new_obs[np.arange(len(vector)), vector] = 1
        return new_obs.flatten()

    def reverse(self, one_hot_vec):
        reshaped = one_hot_vec.reshape((-1, self.n_values))
        return reshaped, reshaped.argmax(axis=1).flatten()


class OneHotWrapper(gym.ObservationWrapper):

    def __init__(self, env, n_values):
        super().__init__(env)
        self.encoder = OneHotEncoder(n_values)

    def observation(self, obs):
        return self.encoder(obs)
