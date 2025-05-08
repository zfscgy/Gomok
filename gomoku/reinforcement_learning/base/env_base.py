from typing import Tuple, Any
import numpy as np


class TwoPlayerEnv:
    def action_space(self) -> list:
        """
        Get the action space (including the invalid actions).
        """
        raise NotImplementedError()

    def get_next_player_id(self) -> int:
        """
        Return: the id of the next player. Usually, the id is 1 or -1.
        """
        raise NotImplementedError()

    def get_last_action(self) -> Any:
        """
        Return: the action of the last player.
        """
        raise NotImplementedError()

    def get_state_for_next_player(self) -> Tuple[np.ndarray, int]:
        """
        Return: the state in the next player's perspective
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

    def trim_history(self):
        """
        Delete the history and only preserve the current state.
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