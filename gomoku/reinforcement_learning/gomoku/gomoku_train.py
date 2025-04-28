import numpy as np
import torch

from gomoku.game.board import GomoBoard

from gomoku.reinforcement_learning.base.player import Game
from gomoku.reinforcement_learning.gomoku.gomoku_env import GomoEnv
from gomoku.reinforcement_learning.gomoku.gomoku_player import get_gomoku_player

from gomoku.reinforcement_learning.base.monte_carlo import play_one_game


class GomokuTrainer:
    def __init__(self, board_size: int, simulations_per_step: int, c_puct: float, device: torch.device):
        self.board_size = board_size
        self.simulations_per_step = simulations_per_step
        self.c_puct = c_puct
        
        self.env = GomoEnv(GomoBoard(board_size))
        self.player = get_gomoku_player(board_size)
        self.game = Game(self.player, self.player, self.env)

        self.model = self.player.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = device

    def play_n_games(self, n: int):
        """
        Play n games.
        """

        data_list = []

        for _ in range(n):
            initial_game = self.game.clone()

            data_list.append(play_one_game(initial_game, self.simulations_per_step, self.c_puct)[0])

        states = np.array([data[0] for data in data_list])
        action_probs = np.array([data[1] for data in data_list])
        values = np.array([data[2] for data in data_list])

        return states, action_probs, values
    
    def train_one_batch(self, states: np.ndarray, action_probs: np.ndarray, values: np.ndarray):
        """
        Train the model.
        """

        states = torch.from_numpy(states).float()[:, None, :, :]
        action_probs = torch.from_numpy(action_probs).float()  # [batch_size, action_size]
        values = torch.from_numpy(values).float()[:, None]  # [batch_size, 1]

        self.optimizer.zero_grad()
        loss = self.model.loss(states, action_probs, values)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def self_play(self, n_games_per_batch: int, n_batches: int):
        """
        Self-play.
        """
        losses = []
        for _ in range(n_batches):
            train_batch = self.play_n_games(n_games_per_batch)
            losses.append(self.train_one_batch(*train_batch))

        return losses


    def save(self, path: str):
        """
        Save the model.
        """
        torch.save(self.model.state_dict(), path)

