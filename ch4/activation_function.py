import numpy as np


def step_function(x: np.ndarray):
    y = x > 0
    return y.astype(int)


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray):
    return np.maximum(0, x)


def identify_function(x: np.ndarray):
    return x


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def sigmoid_grad(x):
    """5章で学ぶ関数。誤差逆伝播法を使う際に必要。
    """
    return (1.0 - sigmoid(x)) * sigmoid(x)
