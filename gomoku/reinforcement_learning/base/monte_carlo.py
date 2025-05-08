from typing import Tuple, List, Any, Callable
from tqdm import tqdm
import time
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


    def get_state(self) -> np.ndarray:
        """
        Get the current state of the game.
        """
        return self.game.env.get_state_for_next_player()

    def simulation(self):
        # Use the NN to estimate the value of the current state
        estimated_value = self.game.get_next_player().value_estimator(self.get_state()[None])[0]

        # When the result is simulated, the value is set to 1 for the current player and -1 for the opponent.
        # The backup node will be updated with the value of the simulated game.
        backup_node = self
        while backup_node is not None:
            estimated_value *= -1
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
        best_child = max(
            node.children, 
            key=lambda child: (-child.total_action_value * 0.5 + 0.5) / (child.visits + 1) + c_puct * (child.prior_mean_value * 0.5 + 0.5) * np.sqrt(np.log(node.visits)) / (child.visits + 1)
        )   # Notice: here the action value is normalized to [0, 1], in order to be consistent with the UCT formula (otherwise, most children will be neglected)
        return select_node(best_child, c_puct)


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
            child_node = MCTSNode(new_game, parent=node)
            node.children.append(child_node)

        children_states = np.array([child.get_state() for child in node.children])
        children_prior_mean_values = player.value_estimator(children_states)
        for child, prior_mean_value in zip(node.children, children_prior_mean_values):
            child.prior_mean_value = prior_mean_value

        node.is_leaf = False
        return True


def alphazero_play_one_game(
        initial_game: Game, simulations_per_step: int, c_puct: float, 
        verbose: bool = True, 
        callback_per_step: Callable[[Game], None] = None
    ) -> Tuple[List[Tuple[Any, np.ndarray, float]], List[MCTSNode]]:
    """
    Play one game using MCTS.
    """
    step_nodes = [MCTSNode(initial_game)]
    print("Start playing one game...")
    t0 = time.time()
    while not step_nodes[-1].game.env.is_end():
        if verbose:
            print(f"Step: {len(step_nodes)}, {time.time() - t0:7.2f}s", end="\r")
        for _ in range(simulations_per_step):
            # Select the node to expand
            selected_node: MCTSNode = select_node(step_nodes[-1], c_puct)
            # Expand the node
            expand(selected_node, step_nodes[-1].game.get_next_player())
            # Simulate the game from the expanded node
            selected_node.simulation()
            # Check if the root node has been fully expanded

        selected_child = max(step_nodes[-1].children, key=lambda child: child.total_action_value / (child.visits+1))
        step_nodes[-1].children.clear()  # Delete the children of the node, since they will not be used again. Otherwise, memory will overflow.
        step_nodes.append(selected_child)

        if callback_per_step is not None:
            callback_per_step(step_nodes[-1].game.env)
        

    winner = step_nodes[-1].game.env.winner()
    if winner is None:
        winner = 0

    state_actionProbs_value: List[Tuple[Any, np.ndarray, float]] = []
    for i in range(0, len(step_nodes) - 1):
        action_probs = np.array([-1] * len(step_nodes[i].game.env.action_space()), dtype=np.float32)  # Invalid steps: action_value = 上个玩家ID（表示当前玩家输了）
        for child in step_nodes[i].children:
            action_probs[np.where(step_nodes[i].game.env.action_space() == child.game.env.get_last_action())[0][0]] = child.total_action_value / (child.visits+1)
        action_probs = np.exp(action_probs) / np.sum(np.exp(action_probs))
        state_actionProbs_value.append((step_nodes[i].get_state(), action_probs, winner))  # [[state, action_probs, value], ...]

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
        action = max(node.children, key=lambda child: child.total_action_value / (child.visits+1)).game.get_last_action()

        game.play(action)
