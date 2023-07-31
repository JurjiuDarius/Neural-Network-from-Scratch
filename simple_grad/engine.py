import numpy as np


class Value:
    def __init__(self, data, children=()) -> None:
        self.data = data
        self.gradient = 0
        self.backward = lambda: None
        self.children = children

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, (self, other))

        def backward():
            self.gradient += out.gradient
            other.gradient += out.gradient

        out.backward = backward
        return out

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, (self, other))

        def backward():
            self.gradient += out.gradient * other.data
            other.gradient += out.gradient * self.data

        out.backward = backward
        return out

    def __pow__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data**other.data, (self, other))

        def backward():
            self.gradient += out.gradient * (self.data ** (other.data - 1)) * other.data
            other.gradient += out.gradient * out.data * np.log(self.data)

        out.backward = backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self))

        def backward():
            self.grad += (out.data > 0) * out.grad

        out.backward = backward

        return out

    def sigmoid(self):
        out = Value(1 / (np.exp(-1) + 1), (self))

        def backward():
            self.grad += out.data * (1 - out.data)

        out.backward = backward

        return out

    def backprop(self, learning_rate):
        self.gradient = 1
        self._backward_recursion(learning_rate)

    def _backward_recursion(self, learning_rate):
        self.backward()
        for child in self.children:
            child.data -= learning_rate * child.gradient
            child._backward_recursion()

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
