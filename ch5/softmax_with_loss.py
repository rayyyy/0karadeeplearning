import numpy as np


def softmax(a: np.ndarray):
    c = np.max(a)
    exp_a = np.exp(a - c)  # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def cross_entropy_error(y: np.ndarray, t: np.ndarray):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.loss = None  # 損失
        self.y = None  # softmaxの出力
        self.t = None  # 教師データ

    def forward(self, x: np.ndarray, t: np.ndarray):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backword(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
