import numpy as np

from gomoku.reinforcement_learning.base.env_base import TwoPlayerEnv

from gomoku.game.board import GomoBoard



class GomoEnv(TwoPlayerEnv):
    def __init__(self, board: GomoBoard):
        self.board = board
    
    def get_current_state(self, player_id: int) -> tuple:
        """
        Return: (state, current_player) The state should be converted to fit the current player's perspective.
        """
        return self.board.board.copy() * player_id
    
    def play(self, action: int) -> bool:
        """
        Play the action. Return True if the action is valid and successfully played.
        """
        x = action // self.board.size
        y = action % self.board.size
        return self.board.play(x, y)
    
    def all_valid_actions(self) -> list:
        """
        Get all valid actions.
        """
        return np.arange(self.board.size * self.board.size)
    
    def is_end(self) -> bool:
        """
        Check if the game has ended.
        """
        return self.board.is_end()
    
    def winner(self) -> int:
        """
        Check if there is a winner.
        """
        return self.board.winner()
