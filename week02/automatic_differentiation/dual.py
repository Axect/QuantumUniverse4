import math

class Dual:
    def __init__(self, x, dx):
        self.x = x
        self.dx = dx

    def __add__(self, other):
        return Dual(self.x + other.x, self.dx + other.dx)

    def __sub__(self, other):
        return Dual(self.x - other.x, self.dx - other.dx)

    def __mul__(self, other):
        return Dual(self.x * other.x, self.x * other.dx + self.dx * other.x)

    def __repr__(self):
        return f"Dual({self.x}, {self.dx})"

def sin(x):
    return Dual(math.sin(x.x), math.cos(x.x) * x.dx)

def cos(x):
    return Dual(math.cos(x.x), -math.sin(x.x) * x.dx)

def exp(x):
    return Dual(math.exp(x.x), math.exp(x.x) * x.dx)

def log(x):
    return Dual(math.log(x.x), x.dx / x.x)
