import threading

from gomoku.game.board import GomoBoard
from gomoku.reinforcement_learning.gomoku.gomoku_env import GomoEnv
from gomoku.reinforcement_learning.gomoku.gomoku_train import GomokuTrainer

from PySide6.QtCore import QThread, Signal, QObject
from PySide6.QtWidgets import QApplication
from gomoku.winui.main_window import GomokuUI


class BoardUpdater(QObject):
    update_signal = Signal()  # Signal to emit x, y coordinates

    def __init__(self, board_ui: GomokuUI):
        super().__init__()
        self.board_ui = board_ui
        self.update_signal.connect(self.board_ui.update_game_state)


app = QApplication([])
board_ui = GomokuUI(GomoBoard())
board_ui.show()

board_updater = BoardUpdater(board_ui)


def callback_per_step(env: GomoEnv):
    board_ui.board.play(*env.board.get_action())
    board_updater.update_signal.emit()

def callback_per_game(env: GomoEnv):
    board_ui.board.reset()
    board_updater.update_signal.emit()


trainer = GomokuTrainer(15, 1000, 5, "cuda", 
                       callback_per_game=callback_per_game,
                       callback_per_step=callback_per_step)


class TrainingThread(QThread):
    def __init__(self, trainer, n_games_per_batch, n_batches):
        super().__init__()
        self.trainer = trainer
        self.n_games_per_batch = n_games_per_batch
        self.n_batches = n_batches

    def run(self):
        self.trainer.self_play(self.n_games_per_batch, self.n_batches)

training_thread = TrainingThread(trainer, 32, 100)
training_thread.start()

print("Training started...")

app.exec()
