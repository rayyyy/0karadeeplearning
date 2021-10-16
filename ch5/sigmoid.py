import numpy as np


class Sigmoid:
    def __init__(self) -> None:
        self.out = None

    def forward(self, x: np.ndarray):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backword(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
