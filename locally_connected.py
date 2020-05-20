import numpy as np
from PIL import Image
import os

def preprocessed_image(image_file):
    image = np.array(image_file)
    image_filtered = np.zeros((16 - 3 + 1, 16 - 3 + 1))
    for i in range(14):
        for j in range(14):
            image_filtered[i, j] = np.sum(image[i:i+3, j:j+3] * SOBEL_X)
    image_filtered = image_filtered.astype('float32') / 255.

    # Padding
    image_padded = np.zeros((17, 17))
    image_padded[1:15, 1:15] = image_filtered

    return image_padded


def train():
    pass


def bit_l2_cost_function(target, output):
    return .5 * np.square(target - output)


def l2_cost_function(target, output):
    return np.mean(bit_l2_cost_function(target, output))


def bit_cost_function(target, output):
    return -target * np.log(output) - (1 - target) * np.log(1 - output)


def cost_function(target, output):
    return np.mean(bit_cost_function(target, output))


def softmax_cross_entropy(target, softmax_class):
    losses = 0
    for i in range(len(target)):
        if softmax_class[i] == 0:
            losses += target[i] * np.log(.00001)
        else:
            losses += target[i] * np.log(softmax_class[i])
    return -losses


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derv(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    sum = np.sum(np.exp(x))
    return np.exp(x) / sum


def leaky_relu(x):
    return np.maximum(.01 * x, x)


def leaky_relu_derv(x):
    if x > 0:
        return 1
    else:
        return .01


def relu(x):
    return np.maximum(0, x)


def relu_derv(x):
    if x is 0:
        return 1
    else:
        return 0


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


SOBEL_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
SOBEL_Y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

# Preprocessing image data
train_images = []
train_labels = []
data_path = './digit data'

for n in range(10):
    for label in range(10):
        fn = str(label) + '.' + str(n) + '.png'
        image_path = os.path.join(data_path, fn)
        image = Image.open(image_path).convert('L')
        train_image = preprocessed_image(image)

        train_images.append(train_image)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

labels_one_hot = []
for i in range(len(train_labels)):
    arr = np.zeros(10)
    arr[train_labels[i]] = 1
    labels_one_hot.append(arr)

train_labels = np.array(labels_one_hot)

# Define parameters
LR = .05
LR_b = .1
epochs = 5000

W_ItoH1 = 2 * np.random.random((3, 3)) - 1
W_H1toH2 = 2 * np.random.random((5, 5)) - 1
W_H2toO = 2 * np.random.random((16, 10)) - 1

b_H1 = 2 * np.random.random((8, 8)) - 1
b_H2 = 2 * np.random.random((4, 4)) - 1
b_O = 2 * np.random.random(10) - 1

# Train
for epoch in range(epochs):
    print('[{:3d}/{} epoch]'.format(epoch+1, epochs), end='    ')
    epoch_acc = 0
    epoch_loss = 0
    for data in range(100):
        image = train_images[data]
        label = train_labels[data]

        # Feedforward
        pre_H1 = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                pre_H1[i, j] = np.sum(image[2*i:2*i+3, 2*j:2*j+3] * W_ItoH1)
        post_H1 = relu(pre_H1) + b_H1

        pre_H2 = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                pre_H2[i, j] = np.sum(post_H1[i:i+5, j:j+5] * W_H1toH2)
        post_H2 = relu(pre_H2) + b_H2

        post_H2_flat = np.reshape(post_H2, 16)
        pre_O = np.zeros(10)
        for i in range(10):
            pre_O[i] = np.sum(post_H2_flat * W_H2toO[:, i])
        post_O = sigmoid(pre_O) + b_O

        softmax_class = softmax(post_O)
        softmax_ce = softmax_cross_entropy(label, softmax_class)
        if np.argmax(softmax_class) == np.argmax(label):
            epoch_acc += 1

        epoch_loss += softmax_ce

        # Backpropagate
        D_post_O = softmax_class - label
        D_pre_O = np.zeros(10)
        for i in range(10):
            D_pre_O[i] = D_post_O[i] * sigmoid_derv(pre_O[i])

        b_O -= LR_b * D_post_O

        W_H2toO_old = W_H2toO
        for i in range(16):
            for j in range(10):
                W_H2toO[i, j] -= LR * D_pre_O[j] * post_H2_flat[i]

        D_post_H2_flat = np.zeros(16)
        for i in range(16):
            for j in range(10):
                D_post_H2_flat[i] = W_H2toO_old[i, j] * D_pre_O[j]
        D_post_H2 = np.reshape(D_post_H2_flat, (4, 4))
        D_pre_H2 = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                D_pre_H2[i, j] = D_post_H2[i, j] * relu_derv(pre_H2[i, j])

        b_H2 -= LR_b * D_post_H2

        W_H1toH2_old = W_H1toH2
        for i in range(5):
            for j in range(5):
                W_H1toH2[i, j] -= LR * np.sum(post_H1[i:i+4, j:j+4] * D_pre_H2)

        W_H1toH2_old_inv = np.flip(W_H1toH2_old)
        D_pre_H2_padded = np.zeros((12, 12))
        D_pre_H2_padded[4:8, 4:8] = D_pre_H2
        for i in range(4, 8):
            for j in range(4, 8):
                D_pre_H2_padded[i][j] = D_pre_H2[i - 4][j - 4]
        D_post_H1 = np.zeros((8, 8))
        D_pre_H1 = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                D_post_H1[i, j] = np.sum(D_pre_H2_padded[i:i+5, j:j+5] * W_H1toH2_old_inv)
                D_pre_H1[i, j] = D_post_H1[i, j] * relu_derv(pre_H1[i, j])

        b_H1 -= LR_b * D_post_H1

        for i in range(3):
            for j in range(3):
                W_ItoH1[i, j] -= LR * np.sum(image[i:i+15:2, j:j+15:2] * D_pre_H1)

    epoch_loss /= 100
    print('Accuracy : {}%, Loss: {}'.format(epoch_acc, epoch_loss))


# Test
while True:
    print('\nType file name : ')
    fn = input()
    test_image_path = os.path.join(data_path, fn)
    test_image = preprocessed_image(Image.open(test_image_path).convert('L'))

    pre_H1 = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            pre_H1[i, j] = np.sum(test_image[2*i:2*i+3, 2*j:2*j+3] * W_ItoH1)
    post_H1 = leaky_relu(pre_H1)

    pre_H2 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            pre_H2[i, j] = np.sum(post_H1[i:i+5, j:j+5] * W_H1toH2)
    post_H2 = leaky_relu(pre_H2)

    post_H2_flat = np.reshape(post_H2, 16)
    pre_O = np.zeros(10)
    post_O = pre_O
    for i in range(10):
        pre_O[i] = np.sum(post_H2_flat * W_H2toO[:, i])
    post_O = sigmoid(pre_O)
