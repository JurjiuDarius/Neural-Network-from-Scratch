import random
from .engine import Value


class Layer:
    def __init__(self, input_dim, output_dim, activation):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = [
            [Value(random.uniform(-1, 1)) for _ in range(input_dim)]
            for __ in range(output_dim)
        ]
        self.biases = [Value(0) for _ in range(output_dim)]
        self.activation = activation

    def forward(self, input):
        output = []
        for i in range(self.output_dim):
            acc = sum(
                (inputi * weightsi for inputi, weightsi in zip(input, self.weights[i])),
                self.biases[i],
            )
            if self.activation == "relu":
                acc = acc.relu()

            output.append(acc)

        return output

    def parameters(self):
        weights = []
        for neuron_weights in self.weights:
            weights += [weight for weight in neuron_weights]
        return weights + self.biases


class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_dim, output_dim, activation=None):
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
            parameter.grad = 0

    def __call__(self, inputs):
        return self.forward(inputs)
