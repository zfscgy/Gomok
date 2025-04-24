from typing import Tuple, Any
import numpy as np


class TwoPlayerEnv:
    def get_current_state(player_id: int = 1) -> Tuple[np.ndarray, int]:
        """
        Return: the state in the player_id's perspective
        """
        raise NotImplementedError()
    
    def play(self, action: Any) -> bool:
        """
        Play the action. Return True if the action is valid and successfully played.
        """
        raise NotImplementedError()
    
    def all_valid_actions(self) -> list:
        """
        Get all valid actions.
        """
        raise NotImplementedError()

    def is_end(self) -> bool:
        """
        Check if the game has ended.
        """
        raise NotImplementedError()
    
    def winner():
        """
        Check if there is a winner.
        """
        raise NotImplementedError()

    def clone(self):
        """
        Clone the environment. This is useful for Monte Carlo Tree Search (MCTS) to simulate future states.
        """
        raise NotImplementedError()

    def to_bytes(self) -> bytes:
        """
        Convert the environment to bytes. This is useful for hashing the state in MCTS.
        """
        raise NotImplementedError()
    
    def from_bytes(self, bs: bytes):
        """
        Load the environment from bytes. This is useful for loading a saved state.
        """
        raise NotImplementedError()