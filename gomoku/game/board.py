import numpy as np

class GomoBoard:
    def __init__(self, board_size: int=15):
        self.board_size = board_size  # 棋盘大小
        self.board = np.zeros((board_size, board_size), dtype=np.int8)  # 0 表示空，1 表示黑棋，-1 表示白棋
        self.current_player = 1  # 1 为黑棋，-1 为白棋
        self.winner = None  # 胜者（1 或 -1），None 表示未分胜负
        self.history = []

    def reset(self):
        """重置棋盘"""
        self.board -= self.board  
        self.current_player = 1
        self.winner = None

    def play(self, x: int, y: int):
        """
        落子操作
        :param x: 落子行
        :param y: 落子列
        :return: 是否成功落子
        """
        if self.board[x, y] != 0 or self.winner is not None:
            return False  # 如果位置已被占用或者已有胜者，不能落子
        self.board[x, y] = self.current_player
        self.history.append(self.board.copy()) 
        
        if self.check_winner(x, y):
            self.winner = self.current_player  # 更新胜者
        self.current_player = - self.current_player  # 切换玩家（1 -> 2 或 2 -> 1）
        return True

    def get_board(self, index: int = -1):
        return self.history[index]
    
    def undo(self):
        if len(self.history) == 0:
            return False
        self.board = self.history.pop()
        self.current_player = - self.current_player
        return True

    def check_winner(self, x: int, y: int):
        """
        检查是否有赢家
        :param x: 最近落子行
        :param y: 最近落子列
        :return: True 表示当前玩家获胜
        """
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 四个方向：横、竖、正斜、反斜
        for dx, dy in directions:
            count = 1  # 包括自己
            for step in range(1, 5):  # 向正方向延伸
                nx, ny = x + step * dx, y + step * dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == self.current_player:
                    count += 1
                else:
                    break
            for step in range(1, 5):  # 向反方向延伸
                nx, ny = x - step * dx, y - step * dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == self.current_player:
                    count += 1
                else:
                    break
            if count >= 5:  # 五子连珠
                return True
        return False
