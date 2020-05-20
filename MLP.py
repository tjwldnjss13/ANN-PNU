import numpy as np
from math import *
from PIL import Image

LEARNING_RATE = 0.00001

in_layer_flat = np.zeros(16 * 16)
h_layer_net = np.zeros(12)
h_layer = np.zeros(12)
out_layer_net = np.zeros(10)
out_layer = np.zeros(10)

w_in_h = np.random.uniform(-0.1, 0.1, (16 * 16, 12))
w_h_out = np.random.uniform(-0.1, 0.1, (12, 10))

d_out_layer = np.zeros(10)
d_out_layer_net = np.zeros(10)
d_h_layer = np.zeros(12)
d_h_layer_net = np.zeros(12)


LABELS = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])


def preprocess(fn):
    global in_layer_flat

    img = Image.open(fn).convert('L')
    img = np.array(img)
    in_layer_flat = np.reshape(img, [16 * 16])

    return in_layer_flat


def feedforward(train, activation):
    global in_layer_flat, h_layer_net, h_layer, out_layer_net, out_layer

    h_layer_net = np.matmul(in_layer_flat, w_in_h)
    h_layer = []
    for net in h_layer_net:
        h_layer.append(activation(net))

    out_layer_net = np.matmul(h_layer, w_h_out)
    out_layer = []
    for net in out_layer_net:
        out_layer.append(activation(net))


def backpropagate(i_target, activation_dif):
    global d_h_layer

    for i in range(10):
        d_out_layer[i] = -(LABELS[i_target][i] - out_layer[i])
        d_out_layer_net[i] = d_out_layer[i] * activation_dif(out_layer_net[i])

    for i in range(12):
        for j in range(10):
            w_h_out[i][j] -= h_layer[i] * d_out_layer_net[j]

    d_h_layer = np.matmul(w_h_out, d_out_layer_net)
    for i in range(12):
        d_h_layer_net[i] = d_h_layer[i] * activation_dif(h_layer_net[i])

    for i in range(16 * 16):
        for j in range(12):
            w_in_h[i][j] -= in_layer_flat[i] * d_h_layer_net[j]


def l2_loss(target):
    pass


def leaky_relu(x):
    return max(0.01 * x, x)


def leaky_relu_dif(x):
    if x > 0:
        return 1
    else:
        return 1 / 100


def sigmoid(x):
    return 1 / (1 + exp(-1 * x))


def sigmoid_dif(x):
    return sigmoid(x) * (1 - sigmoid(x))


def main():
    fn = '../digit data/0.0.png'

    epoch = 10
    i = 1
    while i <= epoch:
        print('[{} epoch]'.format(i))
        img_arr = preprocess(fn)
        feedforward(img_arr, sigmoid)
        backpropagate(0, sigmoid_dif)
        print(out_layer)
        i += 1


if __name__ == '__main__':
    main()
