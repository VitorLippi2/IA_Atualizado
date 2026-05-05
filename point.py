import random

def f(x):
    # y = 0.3x + 0.2
    return 0.3 * x + 0.2

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.bias = 1
        self.label = self._get_label()

    def _get_label(self):
        line_y = f(self.x)
        return 1 if self.y > line_y else -1

    def debug(self):
        print(f"label: {self.label} x {self.x} y {self.y}")