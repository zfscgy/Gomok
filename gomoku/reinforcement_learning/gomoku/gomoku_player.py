import numpy as np
import torch

from gomoku.reinforcement_learning.base.player import IntuitivePlayer
from gomoku.nn.gomoku_model import GomokuModel


def get_gomoku_player(board_size: int) -> IntuitivePlayer:
    """
    Get a Gomoku player.
    """
    model = GomokuModel(board_size, board_size * board_size)
    
    def policy_generator(state: np.ndarray) -> np.ndarray:
        """
        Generate the policy for the given board.
        """

        state = torch.from_numpy(state)[None, None, :, :]
        return torch.softmax(model(state)[0], dim=0).detach().numpy()

    def value_estimator(state: np.ndarray) -> float:
        """
        Estimate the value of the given board.
        """
        state = torch.from_numpy(state)[None, None, :, :]
        return model(state)[1].detach().numpy()

    gomoku_player = IntuitivePlayer(policy_generator, value_estimator)
    gomoku_player.model = model
    return gomoku_player
