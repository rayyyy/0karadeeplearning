import numpy as np


class Relu:
    def __init__(self) -> None:
        self.mask = None

    def forward(self, x: np.ndarray):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backword(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
