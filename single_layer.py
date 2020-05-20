import numpy as np
from PIL import Image
import os
import math

class Perceptron:
    def __init__(self):
        self.W = np.random.uniform(-.1, .1, (16 * 16, 10))
        self.b = np.random.uniform(-.1, .1, 10)

    def exec(self, dataset, learning_rate=0.001, batch_size=10, epochs=20):
        pass


    def dataset(self):
        fp = '../digit data'
        images = []
        labels = []

        for n in range(10):
            for label in range(10):
                fn = str(label) + '.' + str(n) + '.png'
                image = Image.open(os.path.join(fp, fn)).convert('L')
                image = np.array(image)
                images.append(image)
                labels.append(label)

        images = np.array(images)
        labels = Perceptron.one_hot(np.array(labels), 10)

        return [images, labels]

    @staticmethod
    def one_hot(labels, N_label):
        labels_oh = []
        base = np.zeros(N_label)
        for label in labels:
            oh = base
            oh[label] = 1
            labels_oh.append(oh)

        labels_oh = np.array(labels_oh)

        return labels_oh

    def train(self, dataset, learning_rate=0.001, batch_size=10, epochs=20):
        train_images, train_labels = dataset[0], dataset[1]

        for epoch in range(epochs):
            for batch_no in range(math.floor(100 / batch_size)):
                batch_images = train_images[batch_no * batch_size : (batch_no + 1) * batch_size]
                batch_labels = train_labels[batch_no * batch_size : (batch_no + 1) * batch_size]




