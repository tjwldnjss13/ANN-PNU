import numpy as np

def relu(x):
    return np.maximum(x, 0)

def relu_derv(y):
    if y >= 0:
        return 1
    else:
        return 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derv(y):
    return y * (1 - y)

X = np.array([[1, 1, 1, 0, 0, 0],
              [0, 0, 0, 1, 1, 1]])

LABELS = np.array([[1, 0], [0, 1]])

LR = .01
epochs = 20
W_ItoH = np.random.uniform(-.1, .1, (6, 4))
W_HtoO = np.random.uniform(-.1, .1, (4, 2))

for epoch in range(epochs):
    H_pre = np.matmul(X, W_ItoH)
    H_post = relu(H_pre)

    O_pre = np.matmul(H_post, W_HtoO)
    O_post = sigmoid(O_pre)

    J = np.square(np.mean(LABELS - O_post, axis=1))
    print('[{:2%} epoch] Loss : {}'.format(epoch, J))

    D_O_post = O_post - LABELS
    D_P_pre = D_O_post * sigmoid_derv(O_post)


