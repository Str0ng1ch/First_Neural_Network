import numpy as np


np.random.seed(0)
learning_rate = 1


class Layer:
    """ Класс слоя нейронной сети """
    def __init__(self, weights):
        self.weights = weights

    def forward(self, inputs):
        """ Прямой проход """
        self.output = np.dot(inputs, self.weights)


class Activation:
    """ Класс сигмоиндной функции активации """
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-learning_rate * inputs))


class RMS:
    """ Расчет среднеквадратичной ошибки """
    def forward(self, inputs, target):
        self.output = 0.5 * (inputs - target) ** 2


class Change_Error:
    def speed_backwards(self, inputs, target):
        self.speed_output = inputs - target

    def sensitivity_backwards(self, inputs):
        self.output = self.speed_output * inputs * (1 - inputs)


def main():
    inputs = [1, 2]
    activation = Activation()
    rms = RMS()
    change_error = Change_Error()
    layer1 = Layer([[0.1, 0.2], [0.1, 0.2]])
    layer2 = Layer([[0.2], [0.1]])

    layer1.forward(inputs)
    activation.forward(layer1.output)
    print(layer1.output)
    print(activation.output)

    layer2.forward(activation.output)
    activation.forward(layer2.output)
    activation_output = activation.output
    print(layer2.output)
    print(activation.output)

    target = 3
    activation.forward(target)
    rms.forward(activation_output, activation.output)
    print(rms.output)

    change_error.speed_backwards(activation_output, activation.output)
    print(change_error.speed_output)
    change_error.sensitivity_backwards(activation_output)
    print(change_error.output)


if __name__ == "__main__":
    main()
