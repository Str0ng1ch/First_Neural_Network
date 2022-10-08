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


class BackProp:
    def backwards(self, inputs, sensitivity):
        self.output = (sensitivity * inputs).T


def change_speed_error(sensitivity, weights):
    return sensitivity * weights


class ChangeWeights:
    def backwards(self, change_speed, results, inputs):
        self.output = []
        for i in range(len(results)):
            self.output.append(change_speed[i] * results[i] * (1 - results[i]) * inputs)
        self.output = np.array(self.output).T

    def layers(self, weights, EW):
        return weights - EW


def main():
    inputs = [1, 2]
    weight1 = np.array([[0.1, 0.2], [0.1, 0.2]])
    weight2 = np.array([[0.2], [0.1]])
    activation = Activation()
    rms = RMS()
    change_error = Change_Error()
    back_prop = BackProp()
    change_weights = ChangeWeights()

    for _ in range(5000):
        layer1 = Layer(weight1)
        layer2 = Layer(weight2)

        layer1.forward(inputs)
        activation.forward(layer1.output)
        activation1 = activation.output
        print(layer1.output)
        print(activation.output)

        layer2.forward(activation.output)
        activation.forward(layer2.output)
        activation_output = activation.output
        print('Результат', layer2.output)
        print(activation.output)

        target = 3
        activation.forward(target)
        rms.forward(activation_output, activation.output)
        # print(rms.output)

        change_error.speed_backwards(activation_output, activation.output)
        # print(change_error.speed_output)
        change_error.sensitivity_backwards(activation_output)
        # print(change_error.output)

        back_prop.backwards(activation1, change_error.output)
        # print(back_prop.output)

        speed_W1 = change_speed_error(change_error.output, weight2)
        change_weights.backwards(speed_W1, activation1, inputs)
        # print(speed_W1)
        # print(change_weights.output)

        weight1 = change_weights.layers(np.array(weight1), change_weights.output)
        weight2 = change_weights.layers(np.array(weight2), back_prop.output.reshape(2, 1))
        print(weight1)
        print(weight2)
        print()


if __name__ == "__main__":
    main()
