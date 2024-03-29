import numpy as np  # Used solely for the log and exp functions


class Value:
    def __init__(self, data, children=()) -> None:
        self.data = data
        self.grad = 0
        self.backward = lambda: None
        self.children = children

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, (self, other))

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out.backward = backward
        return out

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, (self, other))

        def backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out.backward = backward
        return out

    def __pow__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data**other.data, (self, other))

        def backward():
            self.grad += out.grad * (self.data ** (other.data - 1)) * other.data
            other.grad += out.grad * out.data * np.log(self.data)

        out.backward = backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,))

        def backward():
            self.grad += (out.data > 0) * out.grad

        out.backward = backward

        return out

    def sigmoid(self):
        out = Value(1 / (np.exp(-1) + 1), (self,))

        def backward():
            self.grad += out.data * (1 - out.data)

        out.backward = backward

        return out

    def backprop(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v.backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
