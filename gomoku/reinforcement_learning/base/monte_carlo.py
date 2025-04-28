from typing import Tuple, List, Any

import numpy as np

from gomoku.reinforcement_learning.base.player import Game, IntuitivePlayer


class MCTSNode:
    def __init__(self, game: Game, parent = None, prior_mean_value = None):
        """
        Initialize the MCTS node.
        Args:
            game_base: The game instance but the state could be None
        """
        
        self.game = game

        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.total_action_value = 0.0
        self.prior_mean_value = prior_mean_value

        self.is_leaf = True


    def get_state(self) -> Tuple[np.ndarray, int]:
        """
        Get the current state of the game.
        """
        return self.game.get_state_of_player()

    def simulation(self):
        # Use the NN to estimate the value of the current state
        estimated_value = self.game.get_current_player().value_estimator(self.get_state())
        estimated_value *= self.game.env.get_current_player_id()

        # When the result is simulated, the value is set to 1 for the current player and -1 for the opponent.
        # The backup node will be updated with the value of the simulated game.
        backup_node = self
        while backup_node is not None:
            backup_node.total_action_value += estimated_value
            backup_node.visits += 1
            backup_node = backup_node.parent

        return True


def select_node(node: MCTSNode, c_puct: float) -> MCTSNode:
    """
    Select the node to expand using UCT (Upper Confidence Bound for Trees).
    """
    if node.is_leaf:
        return node
    else:
        best_child = max(node.children, key=lambda child: node.game.env.get_current_player_id() * child.total_action_value / child.visits + c_puct * node.prior_mean_value * np.sqrt(2 * np.log(node.visits) / (child.visits + 1)))
        return select_node(best_child)


def expand(node: MCTSNode, player: IntuitivePlayer):
    """
    Expand the node by generating its children.
    """
    if not node.is_leaf:
        return False
    else:
        for action in node.game.env.all_valid_actions():
            new_game = node.game.clone()
            new_game.env.play(action)
            new_game.env.trim_history()
            child_node = MCTSNode(new_game, parent=node, prior_mean_value=player.value_estimator(new_game.env.get_current_state(new_game.env.get_current_player_id())))
            node.children.append(child_node)
        node.is_leaf = False
        return True


def play_one_game(initial_game: Game, simulations_per_step: int, c_puct: float) -> Tuple[List[Tuple[Any, np.ndarray, float]], List[MCTSNode]]:
    """
    Play one game using MCTS.
    """
    step_nodes = [MCTSNode(initial_game)]
    while step_nodes[-1].game.env.is_end() is False:
        for _ in range(simulations_per_step):
            # Select the node to expand
            selected_node: MCTSNode = select_node(step_nodes[-1], c_puct)
            # Expand the node
            expand(selected_node, step_nodes[-1].game.get_current_player())
            # Simulate the game from the expanded node
            selected_node.simulation()
            # Check if the root node has been fully expanded
            selected_child = max(selected_node.children, key=lambda child: selected_node.game.env.get_current_player_id() * child.total_action_value / child.visits)
            step_nodes.append(selected_child)

    winner = step_nodes[-1].game.env.winner()
    if winner is None:
        winner = 0

    state_actionProbs_value: List[Tuple[Any, np.ndarray, float]] = []
    for i in range(1, len(step_nodes)):
        state_actionProbs_value.append((
            step_nodes[i].get_state(), [child.total_action_value / child.visits for child in step_nodes[i].children], winner
        ))

    return state_actionProbs_value, step_nodes



class MCTSPlayer:
    def __init__(self, intuitive_player: IntuitivePlayer):
        self.intuitive_player = intuitive_player

    def play(self, game: Game, n_simulations: int) -> Tuple[Any, np.ndarray, float]:
        node = MCTSNode(game)

        for _ in range(n_simulations):
            selected_node = select_node(node)
            expand(selected_node, self.intuitive_player)
            selected_node.simulation()
        
        # Find the action with the best average action value
        action = max(node.children, key=lambda child: child.total_action_value / child.visits).game.get_last_action()

        game.play(action)
