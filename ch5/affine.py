import numpy as np


class Affine:
    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backword(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx
