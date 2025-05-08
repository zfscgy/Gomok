import numpy as np

from gomoku.reinforcement_learning.base.env_base import TwoPlayerEnv

from gomoku.game.board import GomoBoard



class GomoEnv(TwoPlayerEnv):
    def __init__(self, board: GomoBoard):
        self.board = board
    

    def action_space(self) -> list:
        """
        Get the action space (including the invalid actions).
        """
        return np.arange(self.board.board_size * self.board.board_size)

    def get_next_player_id(self) -> int:
        """
        Return: the id of the current player
        """
        return self.board.get_player()

    def get_last_action(self) -> int:
        """
        Return: the action of the current player
        """
        x, y = self.board.history[-1][1]
        return self.board.board_size * x + y

    def get_state_for_next_player(self) -> np.ndarray:
        """
        Return: The state converted to fit the next player's perspective.
        """
        return self.board.get_board().copy() * self.get_next_player_id()

    def play(self, action: int) -> bool:
        """
        Play the action. Return True if the action is valid and successfully played.
        """
        x = action // self.board.board_size
        y = action % self.board.board_size
        return self.board.play(x, y)

    def all_valid_actions(self) -> list:
        """
        Get all valid actions. The actions (x, y) is represented as `board_size * x + y`
        """
        return np.argwhere(self.board.get_board().flatten() == 0).flatten()
    
    def is_end(self) -> bool:
        """
        Check if the game has ended.
        """
        return (self.board.winner is not None) or (np.sum(self.board.get_board() == 0) == 0)
    
    def winner(self) -> int:
        """
        Check if there is a winner.
        """
        return self.board.winner
    
    def trim_history(self):
        """
        Delete the history and only preserve the current state.
        """
        self.board.trim_history()


    def clone(self):
        return GomoEnv(self.board.clone())
