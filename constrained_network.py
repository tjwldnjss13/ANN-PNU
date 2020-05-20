import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def one_hot(labels):
    labels_oh = []
    for label in labels:
        arr = np.zeros(10)
        arr[label] = 1
        labels_oh.append(arr)
    labels_oh = np.array(labels_oh)
    return labels_oh


def softmax(x):
    sum = np.sum(np.exp(x))
    return np.exp(x) / sum


def softmax_cross_entropy(target, softmax_class):
    loss = 0
    for i in range(len(target)):
        if softmax_class[i] == 0:
            loss += target[i] * np.log(.00001)
        else:
            loss += target[i] * np.log(softmax_class[i])
    return -loss


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def sigmoid_dif(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return max(0, x)


def relu_dif(x):
    if x > 0:
        return 1
    else:
        return 0


# Parameters
SOBEL_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

W_ItoH1 = np.random.uniform(-.01, .01, (2, 3, 3))
W_H1toH2 = np.random.uniform(-.01, .01, (2, 5, 5))
W_H2toO = np.random.uniform(-.01, .01, (16, 10))

LR = .01
epochs = 5

# Define data path
data_dir = '../digit data'

# Preprocess
train_images = []
train_labels = []

for n in range(10):
    for label in range(10):
        fn = str(label) + '.' + str(n) + '.png'
        image = Image.open(os.path.join(data_dir, fn)).convert('L')
        image = np.array(image)
        train_images.append(image)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
train_labels = one_hot(train_labels)

pre_filter = SOBEL_X
train_images_filtered = []
for img in train_images:
    image_filtered = np.zeros((14, 14))
    for i in range(14):
        for j in range(14):
            for m in range(3):
                for n in range(3):
                    image_filtered[i, j] += img[i+m, j+n] * pre_filter[m, n]
    image_filtered = image_filtered.astype('float32') / 255.
    train_images_filtered.append(image_filtered)

train_images = np.array(train_images_filtered)

image_padded = np.zeros((17, 17))
for i in range(1, 15):
    for j in range(1, 15):
        image_padded[i,j] = image_filtered[i-1,j-1]

image_input = image_padded
activation = sigmoid
activation_dif = sigmoid_dif

# Train
for epoch in range(epochs):
    # Feedforward
    pre_H1 = np.zeros((2, 8, 8))
    post_H1 = np.zeros((2, 8, 8))
    for i in range(8):
        for j in range(8):
            for m in range(3):
                for n in range(3):
                    pre_H1[:,i,j] += image_input[2*i+m,2*j+n] * W_ItoH1[:,m,n]
            post_H1[:,i,j] = activation(pre_H1[:,i,j])

    pre_H2 = np.zeros((4, 4))
    post_H2 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            for m in range(5):
                for n in range(5):
                    pre_H2[i,j] += np.mean(post_H1[:,i+m,j+n] * W_H1toH2[:,m,n])
            # pre_H2[:,i,j] /= 2
    post_H2 = activation(pre_H2)

    post_H2_flattened = np.reshape(post_H2, 16)
    pre_O = np.zeros(10)
    post_O = np.zeros(10)
    for j in range(10):
        for i in range(16):
            pre_O[j] += post_H2_flattened[i] * W_H2toO[i,j]
        post_O[j] = activation(pre_O[j])

    L1 = LABELS[0] - post_O
    L2 = 0.5 * np.square(L1)
    MSE = np.sum(L2) / 10

    # Backpropagate

    # H2 -- O
    D_post_O = -1 * L1
    D_pre_O = activation_dif(pre_O) * D_post_O
    W_H2toO_old = W_H2toO

    for i in range(16):
        for j in range(10):
            W_H2toO[i,j] -= LR * D_pre_O[j] * post_H2_flattened[i]

    D_post_H2_flattened = np.zeros(16)
    for i in range(16):
        for j in range(10):
            D_post_H2_flattened[i] += W_H2toO_old[i,j] * D_pre_O[j]

    # H1 -- H2
    D_post_H2 = np.reshape(D_post_H2_flattened, (4, 4))
    D_pre_H2 = activation_dif(pre_H2) * D_post_H2
    W_H1toH2_old = W_H1toH2

    for i in range(5):
        for j in range(5):
            for m in range(4):
                for n in range(4):
                    W_H1toH2[:,i,j] -= LR * post_H1[:,i+m,j+n] * D_pre_H2[m,n]

    D_pre_H2_padded = np.zeros((12, 12))
    for i in range(4, 8):
        for j in range(4, 8):
            D_pre_H2_padded[i,j] = D_pre_H2[i-4,j-4]

    D_post_H1 = np.zeros((2, 8, 8))
    for i in range(8):
        for j in range(8):
            for m in range(5):
                for n in range(5):
                    D_post_H1[:,i,j] += D_pre_H2_padded[i+m,j+n] * W_H1toH2_old[:,m,n]

    D_pre_H1 = activation_dif(post_H1) * D_post_H2

    # I -- H1


    

