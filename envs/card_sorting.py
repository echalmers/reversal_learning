import random
from typing import SupportsFloat, Any
import numpy as np
from gym.spaces import Discrete
import gymnasium as gym
from gymnasium.core import RenderFrame, ActType, ObsType


class CardSortingEnv(gym.Env):

    def __init__(self, n_classes: int, n_features: int, change_period: int):
        self.n_classes = n_classes
        self.n_features = n_features
        self.change_period = change_period
        self.card = self._draw_card()
        self.match_feature = np.random.randint(n_features)
        self.step_number = 0

        self.observation_space = Discrete(n_features)
        self.action_space = Discrete(n_classes)

    def _draw_card(self):
        return np.random.randint(low=[0] * self.n_features, high=[self.n_classes] * self.n_features)
        # return np.random.choice(self.n_classes, self.n_features, replace=False)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        # determine reward
        correct_act = self.card[self.match_feature]
        reward = 1 if action == correct_act else -1

        # draw next card
        self.card = self._draw_card()

        # rotate match feature
        self.step_number += 1
        if self.step_number % self.change_period == 0:
            self.match_feature += 1
            self.match_feature %= self.n_features
            print(f'change match feature to {self.match_feature}')

        return self.card, reward, False, False, dict()

    def reset(self, *args, **kwars) -> tuple[ObsType, dict[str, Any]]:
        self.card = self._draw_card()
        self.match_feature = np.random.randint(self.n_features)
        self.step_number = 0
        return self.card, dict()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass


if __name__ == '__main__':
    env = CardSortingEnv(n_classes=4, n_features=3, change_period=100)
    card, _ = env.reset()

    while True:
        print(card)
        act = int(input('action'))
        card, r, _, _, _ = env.step(act)
        print(r)

