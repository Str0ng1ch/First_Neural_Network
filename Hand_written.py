import cv2
import numpy as np
import idx2numpy


np.random.seed(0)
TRAIN_IMAGES = idx2numpy.convert_from_file('data/train-images.idx3-ubyte')
TRAIN_LABELS = idx2numpy.convert_from_file('data/train-labels.idx1-ubyte')
TEST_IMAGES = idx2numpy.convert_from_file('data/t10k-images.idx3-ubyte')
TEST_LABELS = idx2numpy.convert_from_file('data/t10k-labels.idx1-ubyte')
learning_rate = 1
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
    # def __init__(self, weights):
    #     self.weights = weights

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


class ChangeError:
    """ Расчет изменения ошибки """
    def speed_output_layer(self, inputs, target):
        self.speed_output = inputs - target

    def sensitivity(self, inputs):
        self.sensitivity_output = self.speed_output * inputs * (1 - inputs)

    def speed_hidden_layer(self, weights):
        self.speed_hidden = self.sensitivity_output * weights


class ChangeWeights:
    """ Изменение синоптических весов """
    def output_layer(self, sensitivity, inputs):
        self.output = []
        for i in range(len(sensitivity[0])):
            print(sensitivity[0][i])
            self.output.append(sensitivity[0][i] * inputs)
        self.output = np.array(self.output).T

    def hidden_layer(self, speed, results, inputs):
        self.output_hidden = []
        for i in range(len(results)):
            self.output_hidden.append(speed[i] * results[i] * (1 - results[i]) * inputs)
        self.output_hidden = np.array(self.output_hidden).T

    def layers(self, weights, EW):
        return weights - EW


def print_digit(array):
    cv2.imshow("Image", array)
    for i in range(28):
        for j in range(28):
            digit = array[i, j]
            print((3 - len(str(float(digit)))) * ' ' + str(float(digit)), end=' ')
        print()
    cv2.waitKey()


def main():
    inputs = TRAIN_IMAGES[0].reshape(1, 784)
    normalization = Normalization()
    normalization.normalization(inputs)
    inputs = normalization.normalization_output
    # inputs = [1, 2]
    # weight1 = np.array([[0.1, 0.2], [0.1, 0.2]])
    # weight2 = np.array([[0.2], [0.1]])
    # layer1 = Layer(weight1)
    # layer2 = Layer(weight2)
    layer1 = Layer(784, 36)
    layer2 = Layer(36, 14)
    activation = Activation()
    rms = RMS()
    change_error = ChangeError()
    change_weight = ChangeWeights()

    for _ in range(1):
        layer1.forward(inputs)
        activation.forward(layer1.output)
        activation1 = activation.output
        print('Выход слоя 1: ', layer1.output)
        print('Выход слоя 1 после активации: ', activation.output)

        layer2.forward(activation.output)
        activation.forward(layer2.output)
        activation_output = activation.output
        print('Выход слоя 2: ', layer2.output)
        print('Выход слоя 2 после активации: ', activation.output)

        target = 3
        activation.forward(target)
        rms.forward(activation_output, activation.output)
        print('Среднеквадратичная ошибка: ', rms.output)

        change_error.speed_output_layer(activation_output, activation.output)
        print('Скорость изменения ошибки выходного слоя: ', change_error.speed_output)
        change_error.sensitivity(activation_output)
        print('Чувствительность изменения ошибки выходного слоя: ', change_error.sensitivity_output)

        change_weight.output_layer(activation1, change_error.sensitivity_output)
        a = change_weight.output
        print('Корректировка матрицы выходного слоя: ', change_weight.output)

        change_error.speed_hidden_layer(layer2.weights)
        print('Скорость изменения ошибки скрытого слоя: ', change_error.speed_hidden)

        change_weight.hidden_layer(change_error.speed_hidden, activation1, inputs)
        print('Расчет коэффициентов скрытого слоя', change_weight.output_hidden)

        weight1 = change_weight.layers(layer1.weights, change_weight.output_hidden)
        print('Измененный вес 1', weight1)
        print(change_weight.output)
        weight2 = change_weight.layers(layer2.weights, np.array(a).reshape(2, 1))
        print('Измененный вес 2', weight2)


if __name__ == "__main__":
    main()
