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


def main():
    inputs = TRAIN_IMAGES[0].reshape(1, 784)
    cv2.imshow("Image", TRAIN_IMAGES[0])
    for i in range(28):
        for j in range(28):
            digit = TRAIN_IMAGES[0, i, j]
            print((3 - len(str(int(digit)))) * ' ' + str(int(digit)), end=' ')
        print()
    activation = Activation()

    layer1 = Layer(784, 36)
    layer1.forward(inputs)
    activation.forward(layer1.output)
    print(activation.output)

    layer2 = Layer(36, 14)
    layer2.forward(activation.output)
    activation.forward(layer2.output)
    print(activation.output)

    layer3 = Layer(14, 10)
    layer3.forward(activation.output)
    activation.forward(layer3.output)

    print(activation.output)
    cv2.waitKey()


if __name__ == "__main__":
    main()
