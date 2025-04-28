import numpy as np

from gomoku.reinforcement_learning.base.env_base import TwoPlayerEnv

from gomoku.game.board import GomoBoard



class GomoEnv(TwoPlayerEnv):
    def __init__(self, board: GomoBoard):
        self.board = board
    
    def get_current_player_id(self) -> int:
        """
        Return: the id of the current player
        """
        return self.board.get_current_player()

    def get_current_action(self) -> tuple:
        """
        Return: the action of the current player
        """
        x, y = self.board.get_current_action()
        return self.board.board_size * x + y

    def get_current_state(self, player_id: int) -> tuple:
        """
        Return: (state, current_player) The state should be converted to fit the current player's perspective.
        """
        return self.board.get_board().copy() * player_id

    def play(self, action: int) -> bool:
        """
        Play the action. Return True if the action is valid and successfully played.
        """
        x = action // self.board.board_size
        y = action % self.board.board_size
        return self.board.play(x, y)

    def all_valid_actions(self) -> list:
        """
        Get all valid actions.
        """
        return np.arange(self.board.board_size * self.board.board_size)
    
    def is_end(self) -> bool:
        """
        Check if the game has ended.
        """
        return (self.board.winner is not None) or (len(self.board.history) >= self.board.board_size * self.board.board_size + 1)
    
    def winner(self) -> int:
        """
        Check if there is a winner.
        """
        return self.board.winner()
    
    def trim_history(self):
        """
        Delete the history and only preserve the current state.
        """
        self.board.trim_history()


    def clone(self):
        return GomoEnv(self.board.clone())
