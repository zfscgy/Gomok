from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QMessageBox, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSlider, QLabel
from PySide6.QtGui import QPen, QBrush, QColor
from PySide6.QtCore import Qt, QRectF

from gomoku.game.board import GomoBoard


class GomokuUI(QMainWindow):
    def __init__(self, board: GomoBoard, board_size_px: int = 600):
        super().__init__()
        self.board = board  # 创建五子棋环境

        self.board_size_px = board_size_px  # 棋盘大小（像素）
        self.cell_size = board_size_px // self.board.board_size  # 每个单元格的大小
        self.display_args = {
            "main_padding": self.board_size_px * 0.05,
            "pawn_size": self.cell_size * 0.9
        }

        # Set global font size
        app = QApplication.instance()
        if app:
            font = app.font()
            font.setPointSize(self.cell_size * 0.3)
            app.setFont(font)

        self.hover_pawn = None

        self.ai_players = []
        self.current_step = -1  # -1 means showing the last step. However, if in the replay mode, it could be from 0...max_step
        self.replay_mode = False

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("五子棋")
        self.setMinimumSize(self.board_size_px * 1.8, self.board_size_px * 1.25)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)  # Set as central widget
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(*[self.display_args["main_padding"]]*4)
        main_layout.setSpacing(20)

        # 棋盘
        self.scene = QGraphicsScene()
        self.board_view = QGraphicsView(self.scene)
        self.board_view.setFixedSize(self.board_size_px * 1.1, self.board_size_px * 1.1)
        self.board_view.setMouseTracking(True)  # 开启鼠标跟踪
        self.draw_board()
        
        main_layout.addWidget(self.board_view)

        # 创建右侧工具栏容器
        toolbar_widget = QWidget()
        toolbar_layout = QVBoxLayout(toolbar_widget)
        toolbar_layout.setSpacing(10)  # Reduced spacing between elements
        toolbar_widget.setFixedWidth(150)
        toolbar_widget.setFixedHeight(self.board_size_px * 0.4)  # Match the board height

        # Undo按钮
        self.undo_button = QPushButton("撤销一步")
        self.undo_button.clicked.connect(self.undo)  # 定义这个方法来处理撤销
        self.undo_button.setMinimumHeight(self.cell_size * 0.6)  # Set minimum height
        toolbar_layout.addWidget(self.undo_button)

        # 重播滑动条
        self.replay_slider = QSlider(Qt.Horizontal)
        self.replay_slider.setMinimum(0)
        self.replay_slider.setMaximum(100)  # 设置为最大步数或稍后动态设置
        self.replay_slider.setValue(0)
        self.replay_slider.setTickPosition(QSlider.TicksBelow)
        self.replay_slider.setTickInterval(1)
        # self.replay_slider.valueChanged.connect(self.handle_replay_slider_changed)
        toolbar_layout.addWidget(QLabel("重播"))
        toolbar_layout.addWidget(self.replay_slider)

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
        if x < 0 or x >= self.board.board_size or y < 0 or y >= self.board.board_size:
            return
        if self.board.current_player in self.ai_players:
            return
        self.put_piece(x, y)

    def handle_move(self, x: int, y: int):
        if self.hover_pawn is not None:
            self.scene.removeItem(self.hover_pawn)
            self.hover_pawn = None  # 移除之前的棋子
        if x < 0 or x >= self.board.board_size or y < 0 or y >= self.board.board_size:
            return

        if self.board.current_player == 1:
            color = QColor(0, 0, 0, 100)
        else:
            color = QColor(255, 255, 255, 100)
        
        self.hover_pawn = self.draw_piece(x, y, color)  # 显示鼠标悬停的棋子

    def put_piece(self, x: int, y: int):
        if self.board.play(x, y):  # 如果落子成功
            self.update_pieces()
            if self.board.winner is not None:  # 检查胜者
                winner_color = "黑棋" if self.board.winner == 1 else "白棋"
                QMessageBox.information(self, "游戏结束", f"{winner_color}获胜！")
                self.board.reset()
                self.scene.clear()
                self.hover_pawn = None
                self.draw_board()


    def undo(self):
        self.board.undo()
        self.update_pieces()


    def update_pieces(self):
        """根据 Env 的状态更新棋子显示"""
        for i in range(self.board.board_size):
            for j in range(self.board.board_size):
                if self.board.get_board()[i, j] == 1:  # 黑棋
                    self.draw_piece(i, j, Qt.black)
                elif self.board.get_board()[i, j] == -1:  # 白棋
                    self.draw_piece(i, j, Qt.white)

    def draw_piece(self, x: float, y: float, color: tuple):
        """绘制棋子"""
        ellipse = QRectF(
            (x + 0.5) * self.cell_size - self.display_args["pawn_size"] / 2, (y + 0.5) * self.cell_size - self.display_args["pawn_size"] / 2,
            self.display_args["pawn_size"], self.display_args["pawn_size"],
        )
        return self.scene.addEllipse(ellipse, QPen(Qt.black), QBrush(color))


if __name__ == "__main__":
    app = QApplication([])
    window = GomokuUI(GomoBoard())
    window.show()
    app.exec()