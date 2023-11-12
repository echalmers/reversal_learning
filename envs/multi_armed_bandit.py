import gymnasium as gym
from gym.spaces import Discrete
import numpy as np
from gymnasium.core import RenderFrame, ActType, ObsType
from typing import SupportsFloat, Any


class VariableGamblingEnv(gym.Env):
    """
        An N-armed bandit that periodically rotates arm reward probabilities
        """

    def __init__(self, p, reward=1, penalty=-1, rotation_interval=None):
        """
        :param p: length-n iterable of arm reward-probabilities
        :param reward: scalar reward delivered on a success
        :param penalty: scalar penalty delivered on a failure
        :param rotation_interval: number of pulls before probabilities rotate
        """
        self.p = np.array(p)
        self.reward = reward
        self.penalty = penalty
        self.rotation_interval = rotation_interval

        self.observation_space = Discrete(1)
        self.action_space = Discrete(len(p))

        self.step_count = 0

    def reset(self, *args, **kwars) -> tuple[ObsType, dict[str, Any]]:
        self.p = np.roll(self.p, 1)
        return np.array([0]), dict()

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1
        if self.rotation_interval is not None and self.step_count % self.rotation_interval == 0:
            self.reset()
            print('rotating probabilities')

        reward = (np.random.random() <= self.p[action]).astype(int)
        reward = reward * (self.reward - self.penalty) + self.penalty
        return np.array([0]), reward, False, False, dict()

    def get_correct_action(self):
        return np.argmax(self.p)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass


if __name__ == '__main__':
    env = VariableGamblingEnv(p=[0.8, 0.1, 0.1], reward=1, penalty=-1, rotation_interval=None)
    _, _ = env.reset()

    while True:
        act = int(input('action'))
        _, r, _, _, _ = env.step(act)
        print(r)