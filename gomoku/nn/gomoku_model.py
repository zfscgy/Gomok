import torch
from torch import nn

class GomokuModel(nn.Module):
    """
    This is a simple model for Gomoku.
    It takes a 2D tensor of the board state and outputs the action probabilities and the value of the board.
    """
    def __init__(self, board_size: int, n_actions: int):
        super(GomokuModel, self).__init__()
        self.board_size = board_size
        self.n_actions = n_actions


        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)


        last_size = ((board_size + 1) // 2 + 1) // 2
        self.fc1 = nn.Linear(128 * last_size * last_size, 1024)
        
        self.output_actions = nn.Linear(1024, n_actions)
        self.output_value = nn.Linear(1024, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, 1, board_size, board_size)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return self.output_actions(x), torch.tanh(self.output_value(x))
