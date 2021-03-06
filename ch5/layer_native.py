class MulLayer:
    def __init__(self) -> None:
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backword(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy
