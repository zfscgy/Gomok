from typing import Callable, Union, Optional
import numpy as np

from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QMessageBox, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSlider, QLabel
from PySide6.QtGui import QPen, QBrush, QColor
from PySide6.QtCore import Qt, QRectF

from gomoku.game.board import GomoBoard


class GomokuUI(QMainWindow):
    def __init__(self, board: GomoBoard, board_size_px: int = 600, ai_play: Callable[[np.ndarray], tuple[int, int]] = None):
        super().__init__()
        self.board = board  # 创建五子棋环境

        self.board_size_px = board_size_px  # 棋盘大小（像素）
        self.cell_size = board_size_px // self.board.board_size  # 每个单元格的大小
        self.display_args = {
            "main_padding": self.board_size_px * 0.02,
            "pawn_size": self.cell_size * 0.9
        }

        # Set global font size
        app = QApplication.instance()
        if app:
            font = app.font()
            font.setPointSize(self.cell_size * 0.3)
            app.setFont(font)

        self.hover_pawn = None
        self.pawns = []
        self.ai_play_func = ai_play

        # This controls the replay mode
        self.replay_mode = False
        self.game_ended = False
        self.current_step = -1  # -1 means showing the last step. However, if in the replay mode, it could be from 0...max_step

        self.init_ui()

    def init_ui(self):
        """
        初始化UI
        * main_layout: 主布局
            * board_view
            * toolbar_layout
                * slider_layout
        """
        self.setWindowTitle("五子棋")
        self.setMinimumSize(self.board_size_px * 1.8, self.board_size_px * 1.05)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)  # Set as central widget
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(*[self.display_args["main_padding"]]*4)
        main_layout.setSpacing(20)

        # 棋盘
        self.scene = QGraphicsScene()
        self.board_view = QGraphicsView(self.scene)
        self.board_view.setFixedSize(self.board_size_px * 1.04, self.board_size_px * 1.04)
        self.board_view.setMouseTracking(True)  # 开启鼠标跟踪
        self.draw_board()
        
        main_layout.addWidget(self.board_view)

        # 创建右侧工具栏容器
        toolbar_widget = QWidget()
        toolbar_layout = QVBoxLayout(toolbar_widget)
        toolbar_layout.setSpacing(10)  # Reduced spacing between elements
        toolbar_widget.setFixedWidth(self.board_size_px * 0.7)
        toolbar_widget.setFixedHeight(self.board_size_px * 0.4)  # Match the board height


        control_layout = QHBoxLayout()
        # Undo按钮（）
        self.undo_button = QPushButton("撤销一步")
        self.undo_button.setMaximumWidth(self.cell_size * 2)
        self.undo_button.clicked.connect(self.undo)  # 定义这个方法来处理撤销
        self.undo_button.setMinimumHeight(self.cell_size * 0.6)  # Set minimum height
        control_layout.addWidget(self.undo_button)

        self.ai_play_button = QPushButton("AI落子")
        self.ai_play_button.setMaximumWidth(self.cell_size * 2)
        self.ai_play_button.setMinimumHeight(self.cell_size * 0.6)
        self.ai_play_button.clicked.connect(self.ai_play)
        control_layout.addWidget(self.ai_play_button)

        toolbar_layout.addLayout(control_layout)


        #  新的一行
        mode_layout = QHBoxLayout()
        mode_layout.setContentsMargins(*[self.display_args["main_padding"]]*4)
        # 当前模式
        self.mode_label = QLabel("当前模式：对战模式")
        self.mode_label.setMaximumWidth(self.cell_size * 4)
        self.mode_label.setMinimumHeight(self.cell_size * 0.6)
        mode_layout.addWidget(self.mode_label)


        # 模式切换按钮
        self.mode_button = QPushButton("切换到回放模式")
        self.mode_button.setMaximumWidth(self.cell_size * 4)
        self.mode_button.setMinimumHeight(self.cell_size * 0.6)
        self.mode_button.clicked.connect(self.switch_mode)
        mode_layout.addWidget(self.mode_button)

        toolbar_layout.addLayout(mode_layout)

        # 重播滑动条
        self.replay_slider = QSlider(Qt.Horizontal)
        self.replay_slider.setMinimum(0)
        self.replay_slider.setMaximum(1)  # 一开始假设只有一步
        self.replay_slider.setValue(1)   # 一开始设置为最大处
        self.replay_slider.setTickPosition(QSlider.TicksBelow)
        self.replay_slider.setTickInterval(1)
        self.replay_slider.setEnabled(False)
        self.replay_slider.valueChanged.connect(self.handle_replay_slider_changed)
        toolbar_layout.addWidget(self.replay_slider)
        


        slider_layout = QHBoxLayout()
        self.replay_back_button = QPushButton("<")
        self.replay_back_button.setMaximumWidth(self.cell_size * 0.6)
        self.replay_back_button.setMinimumHeight(self.cell_size * 0.6)
        self.replay_back_button.setEnabled(False)
        slider_layout.addWidget(self.replay_back_button)
        self.replay_back_button.clicked.connect(self.replay_back)

        self.replay_forward_button = QPushButton(">")
        self.replay_forward_button.setMaximumWidth(self.cell_size * 0.6)
        self.replay_forward_button.setMinimumHeight(self.cell_size * 0.6)
        self.replay_forward_button.setEnabled(False)
        slider_layout.addWidget(self.replay_forward_button)
        self.replay_forward_button.clicked.connect(self.replay_forward)

        self.replay_step_label = QLabel(f"当前步数：{self.current_step} / {len(self.board.history) - 1}")
        self.replay_step_label.setMaximumWidth(self.cell_size * 4)
        self.replay_step_label.setMinimumHeight(self.cell_size * 0.6)
        slider_layout.addWidget(self.replay_step_label)

        toolbar_layout.addLayout(slider_layout)

        # Add stretch to push elements to the top
        toolbar_layout.addStretch()

        main_layout.addWidget(toolbar_widget)

    def draw_board(self):
        """绘制棋盘背景和网格线"""
        board_size = self.board.board_size

        # 棋盘背景
        self.scene.setBackgroundBrush(QBrush(QColor(210, 180, 140)))  # 棕色背景

        # 绘制网格线
        pen = QPen(Qt.black, 1)
        for i in range(board_size):
            # 横线
            self.scene.addLine(0,  (i + 0.5) * self.cell_size, self.board_size_px, (i + 0.5) * self.cell_size, pen)
            # 竖线
            self.scene.addLine((i + 0.5) * self.cell_size, 0, (i + 0.5) * self.cell_size, self.board_size_px, pen)

        def get_board_col_row(x: float, y: float):
            """获取单元格的中心坐标"""
            col = x / self.cell_size - 0.5
            row = y / self.cell_size - 0.5
            return int(round(col)), int(round(row))

        self.scene.mousePressEvent = lambda event: self.handle_click(*get_board_col_row(event.scenePos().x(), event.scenePos().y()))
        self.scene.mouseMoveEvent = lambda event: self.handle_move(*get_board_col_row(event.scenePos().x(), event.scenePos().y()))

    def handle_click(self, x: int, y: int):
        """处理用户点击事件"""
        if self.replay_mode:
            return
        if x < 0 or x >= self.board.board_size or y < 0 or y >= self.board.board_size:
            return
        self.put_piece(x, y)

    def handle_move(self, x: int, y: int):
        if self.hover_pawn is not None:
            self.scene.removeItem(self.hover_pawn)
            self.hover_pawn = None  # 移除之前的棋子
        if x < 0 or x >= self.board.board_size or y < 0 or y >= self.board.board_size:
            return

        if self.board.get_player() == -1:  # 注意到current_player表示的是上一步的，不是这一步的
            color = QColor(0, 0, 0, 100)
        else:
            color = QColor(255, 255, 255, 100)
        
        self.hover_pawn = self.draw_piece(x, y, color)  # 显示鼠标悬停的棋子

    def put_piece(self, x: int, y: int):
        if self.board.winner is not None:
            pass
        elif self.board.play(x, y):  # 如果落子成功
            self.update_game_state()

        if self.board.winner is not None:  # 检查胜者
            winner_color = "黑棋" if self.board.winner == 1 else "白棋"
            QMessageBox.information(self, "游戏结束", f"{winner_color}获胜！")
            self.switch_mode()


    def undo(self):
        self.board.undo()
        self.update_game_state()
    
    def ai_play(self):
        if self.ai_play_func is None:
            return
        x, y = self.ai_play_func(self.board.get_board())
        self.put_piece(x, y)

    def switch_mode(self):
        # 切换到回放模式
        if not self.replay_mode:
            self.replay_mode = True
            self.mode_button.setText("切换到对战模式")
            self.mode_label.setText("当前模式：回放模式")
            self.undo_button.setEnabled(False)
            self.replay_slider.setEnabled(True)
            self.replay_slider.setMaximum(len(self.board.history) - 1)
            self.replay_slider.setValue(len(self.board.history) - 1)
            self.replay_back_button.setEnabled(True)
            self.replay_forward_button.setEnabled(True)
            self.current_step = len(self.board.history) - 1

        # 切换到对战模式（从当前回放处开始）
        else:
            self.replay_mode = False
            self.mode_button.setText("切换到回放模式")
            self.mode_label.setText("当前模式：对战模式")
            self.undo_button.setEnabled(True)
            self.replay_slider.setEnabled(False)
            self.replay_back_button.setEnabled(False)
            self.replay_forward_button.setEnabled(False)
            self.board.jump_to(self.current_step)
            
            self.current_step = -1

    def replay_back(self):
        if self.current_step == -1:
            self.current_step = len(self.board.history) - 1
        if self.current_step > 0:
            self.current_step -= 1
        self.replay_slider.setValue(self.current_step)

    def replay_forward(self):
        if self.current_step == -1:
            self.current_step = len(self.board.history) - 1
        if self.current_step < len(self.board.history) - 1:
            self.current_step += 1
        self.replay_slider.setValue(self.current_step)
    
    def handle_replay_slider_changed(self):
        self.current_step = self.replay_slider.value()

        self.update_game_state()


    def update_game_state(self):
        """根据 Env 的状态更新棋子和其他UI"""
        for pawn in self.pawns:
            self.scene.removeItem(pawn)
        self.pawns.clear()

        self.replay_step_label.setText(f"当前步数：{self.current_step} / {len(self.board.history) - 1}")
        board = self.board.get_board(self.current_step)
        
        # Get the last 5 moves
        last_5_moves = []
        if self.current_step == -1:
            last_5_moves = self.board.history[1:][-5:]
        else:
            last_5_moves = self.board.history[1:][max(0, self.current_step-4):self.current_step+1]
        
        # Create a dictionary mapping coordinates to move numbers
        move_numbers = dict()
        for idx, (_, (x, y), _) in enumerate(last_5_moves):
            move_numbers[(x, y)] = len(self.board.history) - len(last_5_moves) + idx

        for i in range(self.board.board_size):
            for j in range(self.board.board_size):
                if board[i, j] == 1:  # 黑棋
                    self.pawns.append(self.draw_piece(i, j, Qt.black, move_numbers.get((i, j))))
                elif board[i, j] == -1:  # 白棋
                    self.pawns.append(self.draw_piece(i, j, Qt.white, move_numbers.get((i, j))))

    def draw_piece(self, x: float, y: float, color: tuple, move_number: Optional[int] = None):
        """绘制棋子"""
        ellipse = QRectF(
            (x + 0.5) * self.cell_size - self.display_args["pawn_size"] / 2, 
            (y + 0.5) * self.cell_size - self.display_args["pawn_size"] / 2,
            self.display_args["pawn_size"], 
            self.display_args["pawn_size"],
        )
        piece = self.scene.addEllipse(ellipse, QPen(Qt.black), QBrush(color))
        
        if move_number is not None:
            text = self.scene.addText(str(move_number))
            text.setDefaultTextColor(Qt.red)
            font = text.font()
            font.setPointSize(self.cell_size * 0.3)
            text.setFont(font)
            
            # Center the text on the piece
            text_width = text.boundingRect().width()
            text_height = text.boundingRect().height()
            text_x = (x + 0.5) * self.cell_size - text_width / 2
            text_y = (y + 0.5) * self.cell_size - text_height / 2
            text.setPos(text_x, text_y)
            
            # Group the piece and text together
            group = self.scene.createItemGroup([piece, text])
            return group
 
        return piece


if __name__ == "__main__":
    app = QApplication([])
    window = GomokuUI(GomoBoard())
    window.show()
    app.exec()