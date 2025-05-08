from typing import Callable

import numpy as np

from gomoku.reinforcement_learning.base.env_base import TwoPlayerEnv



class IntuitivePlayer:
    def __init__(self, policy_generator: Callable, value_estimator: Callable):
        """
        policy_generator: numpy [batch_size, state...] -> [batch_size, action_size]
        value_estimator: numpy [batch_size, state...] -> [batch_size, 1]
        """
        self.policy_generator = policy_generator
        self.value_estimator = value_estimator

    def play(self, env: TwoPlayerEnv):
        policy = self.policy_generator(env)
        action = np.argmax(policy)
        env.play(action)
        return action


class Game:
    def __init__(self,player1: IntuitivePlayer, player2: IntuitivePlayer, env: TwoPlayerEnv = None):
        self.env: TwoPlayerEnv = env
        self.players = [player2, player1]

    def get_next_player(self) -> IntuitivePlayer:
        return self.players[int((self.env.get_next_player_id() + 1)/2)]

    def play(self, action = None):
        """
        If action is not assigned, use the current player's policy function to get the best action
        """
        if action is None:
            action = self.get_next_player().play(self.env)
        self.env.play(action)
        return action

    def clone(self):
        return Game(self.players[0], self.players[1], self.env.clone())
