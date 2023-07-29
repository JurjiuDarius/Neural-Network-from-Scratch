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

    def backprop(self):
        self.gradient = 1
        self._backward_recursion()

    def _backward_recursion(self):
        self.backward()
        for child in self.children:
            child.data -= self.gradient
            child._backward_recursion()
