# cnn/model.py
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.block(x)

class CNNEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(3, 64, 3, 1, 1)
        self.body = nn.Sequential(*[ResBlock(64) for _ in range(6)])
        self.tail = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x.clamp(0, 1)