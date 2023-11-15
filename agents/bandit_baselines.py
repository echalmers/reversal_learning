from collections import namedtuple

# from learners.abstract import ReinforcementLearner
import numpy as np
from bandits.agent import Agent, GradientAgent, BetaAgent
from bandits.policy import UCBPolicy, SoftmaxPolicy, EpsilonGreedyPolicy
from bandits.bandit import BernoulliBandit, MultiArmedBandit


class BGalbraithLearner:  #(ReinforcementLearner):

    def __init__(self, agent):
        self.agent = agent

    def action_probabilities(self):
        p = np.exp(self.agent.value_estimates) / np.sum(np.exp(self.agent.value_estimates))
        return {i: p[i] for i in range(len(p))}

    def select_action(self, state):
        return self.agent.choose()

    def update(self, state, action, reward, new_state, done):
        self.agent.observe(reward)


class UCBLearner(BGalbraithLearner):

    def __init__(self, n, c):
        super().__init__(Agent(MultiArmedBandit(k=n), UCBPolicy(c)))


class PerfectInfoAgent(BGalbraithLearner):

    def __init__(self, n, c, rotation_interval):
        super().__init__(Agent(MultiArmedBandit(k=n), UCBPolicy(c)))
        self.step_count = 0
        self.rotation_interval = rotation_interval

    def update(self, state, action, reward, new_state, done):
        self.step_count += 1
        if self.step_count % self.rotation_interval == 0:
            self.agent.reset()
        return super().update(state, action, reward, new_state, done)

class GradientLearner(BGalbraithLearner):

    def __init__(self, n, alpha, baseline):
        super().__init__(GradientAgent(MultiArmedBandit(k=n), EpsilonGreedyPolicy(0.1), alpha=alpha, baseline=baseline))

