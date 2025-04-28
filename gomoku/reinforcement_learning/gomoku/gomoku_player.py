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
        device = next(model.parameters()).device
        state = torch.from_numpy(state)[None, None, :, :].float().to(device)
        action_probs = torch.softmax(model(state)[0], dim=0) * (state == 0).float()  # (state == 0) 表示未落子位置
        return action_probs.flatten().detach().numpy()

    def value_estimator(state: np.ndarray) -> float:
        """
        Estimate the value of the given board.
        """
        device = next(model.parameters()).device
        state = torch.from_numpy(state)[None, None, :, :].float().to(device)
        return model(state)[1].detach().numpy()

    gomoku_player = IntuitivePlayer(policy_generator, value_estimator)
    gomoku_player.model = model
    return gomoku_player
