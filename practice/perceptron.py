def AND(x1: float, x2: float):
    w1, w2, theta = 1.0, 1.0, 1.0
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
