import random
from .engine import Value
import numpy as np


class Layer:
    def __init__(self, input_dim, output_dim, activation):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = [[Value(random.uniform(-1, 1))] * input_dim] * output_dim
        self.biases = [Value(random.uniform(-1, 1))] * output_dim
        self.activation = activation

    def forward(self, input):
        output = []
        for i in range(self.output_dim):
            acc = 0
            for j in range(self.input_dim):
                acc += input[j] * self.weights[i][j]

            a = 0
            if self.activation == "relu":
                a = acc.relu()
            else:
                a = acc.sigmoid()
            output.append(a)

        return output

    def parameters(self):
        weights = []
        for neuron_weights in self.weights:
            weights += [weight for weight in neuron_weights]
        return weights + self.biases


class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_dim, output_dim, activation):
        self.layers.append(Layer(input_dim, output_dim, activation))

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()

        return params

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.gradient = 0

    def __call__(self, inputs):
        return self.forward(inputs)
