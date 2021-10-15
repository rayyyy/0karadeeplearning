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


def softmax(a: np.ndarray):
    c = np.max(a)
    exp_a = np.exp(a - c)  # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
