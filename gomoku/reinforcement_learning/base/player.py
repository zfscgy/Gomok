from typing import Callable

import numpy as np

from gomoku.reinforcement_learning.base.env_base import TwoPlayerEnv



class IntuitivePlayer:
    def __init__(self, policy_generator: Callable, value_estimator: Callable):
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
        self.current_player_id = 1


    def get_current_player(self) -> IntuitivePlayer:
        return self.players[(self.current_player_id + 1)/2]


    def get_state_of_player(self):
        return self.env.get_current_state(self.current_player_id)

    def play(self, action = None):
        """
        If action is not assigned, use the current player's policy function to get the best action
        """
        if action is None:
            action = self.get_current_player().play(self.env)
        self.env.play(action)
        self.current_player_id *= -1
        return action

    def clone(self):
        return Game(self.env.clone(), self.players[0], self.players[1])
