import cv2
import numpy as np
import idx2numpy


np.random.seed(0)
TRAIN_IMAGES = idx2numpy.convert_from_file('data/train-images.idx3-ubyte')
TRAIN_LABELS = idx2numpy.convert_from_file('data/train-labels.idx1-ubyte')
TEST_IMAGES = idx2numpy.convert_from_file('data/t10k-images.idx3-ubyte')
TEST_LABELS = idx2numpy.convert_from_file('data/t10k-labels.idx1-ubyte')
learning_rate = 0.1
SIZE = 28


class Normalization:
    """ Класс нормализации данных """
    def normalization(self, inputs):
        minimum = np.min(inputs)
        max_minus_min = np.max(inputs) - minimum
        self.normalization_output = (inputs - minimum) / max_minus_min

    def denormalization(self, inputs):
        minimum = np.min(inputs)
        max_minus_min = np.max(inputs) - minimum
        self.denormalization_output = inputs * max_minus_min + minimum


class Layer:
    """ Класс слоя нейронной сети """
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

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


def print_digit(array):
    cv2.imshow("Image", array)
    for i in range(28):
        for j in range(28):
            digit = array[i, j]
            print((3 - len(str(float(digit)))) * ' ' + str(float(digit)), end=' ')
        print()


def main():
    inputs = TRAIN_IMAGES[0].reshape(1, 784)
    normalization = Normalization()
    input_layer = Layer(784, 36)
    hidden_layer1 = Layer(36, 14)
    output_layer = Layer(14, 10)
    activation = Activation()
    rms = RMS()

    normalization.normalization(inputs)
    inputs = normalization.normalization_output

    for _ in range(5):
        input_layer.forward(inputs)
        activation.forward(input_layer.output)
        input_activation = activation.output
        print(activation.output)

        hidden_layer1.forward(input_activation)
        activation.forward(hidden_layer1.output)
        hidden1_activation = activation.output
        print(activation.output)

        output_layer.forward(hidden1_activation)
        activation.forward(output_layer.output)
        output_activation = activation.output

        print(output_activation)

        target = np.array([0] * 10)
        target[TRAIN_LABELS[0] - 1] = 1
        rms.forward(output_activation, target)
        print(rms.output)


    normalization.denormalization(output_activation)
    print(normalization.denormalization_output)
    cv2.waitKey()


if __name__ == "__main__":
    main()
