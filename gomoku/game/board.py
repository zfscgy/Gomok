import numpy as np

class GomoBoard:
    def __init__(self, board_size: int=15):
        self.board_size = board_size  # 棋盘大小
        self.winner = None  # 胜者（1 或 -1），None 表示未分胜负
        self.history = [(np.zeros((board_size, board_size), dtype=np.int8), None, -1)]  # [棋盘状态、上个落子、上个落子玩家]

    def reset(self):
        """重置棋盘"""
        self.history = self.history[:1]
        self.winner = None

    def play(self, x: int, y: int):
        """
        落子操作
        :param x: 落子行
        :param y: 落子列
        :return: 是否成功落子
        """
        board, _, last_player = self.history[-1]
        board = board.copy()
        current_player = -last_player
        if board[x, y] != 0 or self.winner is not None:
            return False  # 如果位置已被占用或者已有胜者，不能落子
        board[x, y] = current_player
        self.history.append((board, (x, y), current_player))
        self.check_game_ended()
        return True

    def check_game_ended(self):
        self.winner = None
        board, action, player = self.history[-1]
        if action is None:
            return False
        else:
            x, y = action
            directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 四个方向：横、竖、正斜、反斜
            for dx, dy in directions:
                count = 1  # 包括自己
                for step in range(1, 5):  # 向正方向延伸
                    nx, ny = x + step * dx, y + step * dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.get_board()[nx, ny] == self.get_player():
                        count += 1
                    else:
                        break
                for step in range(1, 5):  # 向反方向延伸
                    nx, ny = x - step * dx, y - step * dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.get_board()[nx, ny] == self.get_player():
                        count += 1
                    else:
                        break
                if count >= 5:  # 五子连珠
                    self.winner = player
                    return True
            return False

    def get_board(self, index: int = -1):
        return self.history[index][0]

    def get_player(self, index: int = -1):
        return self.history[index][2]
    
    def get_action(self, index: int = -1):
        return self.history[index][1]

    def undo(self):
        if len(self.history) == 1:
            return False
        self.history.pop()
        return True

    def jump_to(self, index: int):
        self.history = self.history[:index+1]
        self.check_game_ended()
    
    def trim_history(self):
        self.history = self.history[-1:]

    def clone(self):
        board = GomoBoard(self.board_size)
        board.history = [(board.copy(), action, player) for board, action, player in self.history]
        board.winner = self.winner
        return board
