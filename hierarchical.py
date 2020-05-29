import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from filters import *


class HierarchicalNet:
    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    def __init__(self, lr=.001, epochs=10):
        self.lr = lr
        self.lr_b = lr * 2
        self.epochs = epochs

        self.W_ItoH1 = 2 * np.random.random((3, 3)) - 1
        self.W_H1toH2 = 2 * np.random.random((5, 5)) - 1
        self.W_H2toO = 2 * np.random.random((16, 10)) - 1

        self.b_H1 = 2 * np.random.random((8, 8)) - 1
        self.b_H2 = 2 * np.random.random((4, 4)) - 1
        self.b_O = 2 * np.random.random(10) - 1
        # b_H1, b_H2, b_O = 1, 1, 1

    def exec_all(self, fp, filter=None):
        images, labels = self.dataset(fp, filter)

        train_images, train_labels = images[:70], labels[:70]
        valid_images, valid_labels = images[70:85], labels[70:85]
        test_images, test_labels = images[85:], labels[85:]

        self.train([train_images, train_labels], [valid_images, valid_labels])
        self.test([test_images, test_labels])
        self.visualize()

    def dataset(self, fp, filter):
        images = []
        labels = []

        for n in range(10):
            for label in range(10):
                fn = str(label) + '.' + str(n) + '.png'
                image_path = os.path.join(fp, fn)
                image = Image.open(image_path).convert('L')
                image = self.preprocessed_image(image, filter)
                images.append(image)
                labels.append(label)

        images = np.array(images)
        labels = np.array(labels)

        labels_one_hot = []
        for i in range(len(labels)):
            arr = np.zeros(10)
            arr[labels[i]] = 1
            labels_one_hot.append(arr)

        labels = np.array(labels_one_hot)

        return images, labels

    @staticmethod
    def preprocessed_image(image_file, filter):
        image = np.array(image_file)
        image_padded = np.zeros((17, 17))
        if filter is None:
            image = image.astype('float32') / 255
            image_padded[:17, :17] = image
        else:
            f_size = len(filter)
            if_size = 16 - f_size + 1
            image_filtered = np.zeros((if_size, if_size))
            for i in range(if_size):
                for j in range(if_size):
                    image_filtered[i, j] = np.sum(image[i:i+f_size, j:j+f_size] * filter)
            image_filtered = image_filtered.astype('float32') / 255
            image_padded[1:1+if_size, 1:1+if_size] = image_filtered

        return image_padded

    def feedforward(self, image):
        pre_H1 = np.zeros((2, 8, 8))
        for k1 in range(2):
            for i in range(8):
                for j in range(8):
                    pre_H1[k1, i, j] = np.sum(image[2 * i:2 * i + 3, 2 * j:2 * j + 3] *self.W_ItoH1)
        post_H1 = HierarchicalNet.relu(pre_H1) + self.b_H1

        pre_H2 = np.zeros((4, 4, 4))
        for k1 in range(2):
            for k2 in range(4):
                for i in range(4):
                    for j in range(4):
                        pre_H2[k2, i, j] += np.sum(post_H1[k1, i:i + 5, j:j + 5] * self.W_H1toH2)
        post_H2 = HierarchicalNet.relu(pre_H2) + self.b_H2

        post_H2_flattened = np.reshape(post_H2, 4 * 4 * 4)
        pre_O = np.matmul(post_H2_flattened, self.W_H2toO)
        post_O = HierarchicalNet.sigmoid(pre_O) + self.b_O

        return post_O

    def train(self, train_dataset, valid_dataset):
        train_images, train_labels = train_dataset[0], train_dataset[1]
        valid_images, valid_labels = valid_dataset[0], valid_dataset[1]

        for epoch in range(self.epochs):
            self.epoch_list.append(epoch)
            print('[{}/{} epoch]'.format(epoch + 1, self.epochs), end='    ')
            train_acc = 0
            train_loss = 0

            # Feedforward
            for train_idx in range(len(train_images)):
                image = train_images[train_idx]
                label = train_labels[train_idx]

                pre_H1 = np.zeros((2, 8, 8))
                for k1 in range(2):
                    for i in range(8):
                        for j in range(8):
                            pre_H1[k1, i, j] = np.sum(image[2 * i:2 * i + 3, 2 * j:2 * j + 3] * self.W_ItoH1)
                post_H1 = HierarchicalNet.relu(pre_H1) + self.b_H1

                pre_H2 = np.zeros((4, 4, 4))
                for k1 in range(2):
                    for k2 in range(4):
                        for i in range(4):
                            for j in range(4):
                                pre_H2[k2, i, j] += np.sum(post_H1[k1, i:i + 5, j:j + 5] * self.W_H1toH2)
                post_H2 = HierarchicalNet.relu(pre_H2) + self.b_H2

                post_H2_flattened = np.reshape(post_H2, 4 * 4 * 4)
                pre_O = np.matmul(post_H2_flattened, self.W_H2toO)
                post_O = HierarchicalNet.sigmoid(pre_O) + self.b_O

                softmax_class = HierarchicalNet.softmax(post_O)
                softmax_ce = HierarchicalNet.softmax_cross_entropy(label, softmax_class)
                if np.argmax(softmax_class) == np.argmax(label):
                    train_acc += 1
                train_loss += softmax_ce

                # Backpropagate
                # O
                D_post_O = softmax_class - label
                D_pre_O = D_post_O * HierarchicalNet.sigmoid_derv(pre_O)

                # Bias (O)
                self.b_O -= self.lr_b * D_post_O

                # Weight (H2 -- O)
                W_H2toO_old = self.W_H2toO
                for i in range(4 * 4 * 4):
                    for j in range(10):
                        self.W_H2toO[i, j] -= self.lr * D_pre_O[j] * post_H2_flattened[i]

                # H2
                D_post_H2_flattened = np.zeros(4 * 4 * 4)
                for i in range(4 * 4 * 4):
                    for j in range(10):
                        D_post_H2_flattened[i] += W_H2toO_old[i, j] * D_pre_O[j]

                D_post_H2 = np.reshape(D_post_H2_flattened, (4, 4, 4))
                D_pre_H2 = np.zeros((4, 4, 4))
                for k2 in range(4):
                    for i in range(4):
                        for j in range(4):
                            D_pre_H2[k2, i, j] = D_post_H2[k2, i, j] * HierarchicalNet.relu_derv(pre_H2[k2, i, j])

                # Bias (H2)
                self.b_H2 -= self.lr_b * D_post_H2

                # Weight (H1 -- H2)
                W_H1toH2_old = self.W_H1toH2
                for k1 in range(2):
                    for k2 in range(4):
                        for i in range(5):
                            for j in range(5):
                                self.W_H1toH2[i, j] -= self.lr * np.sum(post_H1[k1, i:i + 4, j:j + 4] * D_pre_H2[k2])

                # H1
                W_H1toH2_old_inv = np.flip(W_H1toH2_old)
                D_pre_H2_padded = np.zeros((4, 12, 12))
                D_pre_H2_padded[:, 4:8, 4:8] = D_pre_H2[:]
                # for i in range(4, 8):
                #     for j in range(4, 8):
                #         D_pre_H2_padded[i,j] = D_pre_H2[i-4, j-4]
                D_post_H1 = np.zeros((2, 8, 8))
                D_pre_H1 = np.zeros((2, 8, 8))
                for k1 in range(2):
                    for k2 in range(4):
                        for i in range(8):
                            for j in range(8):
                                D_post_H1[k1, i, j] += np.sum(D_pre_H2_padded[k2, i:i + 5, j:j + 5] * W_H1toH2_old_inv)

                for k1 in range(2):
                    for i in range(8):
                        for j in range(8):
                            D_pre_H1[k1, i, j] = D_post_H1[k1, i, j] * HierarchicalNet.relu_derv(pre_H1[k1, i, j])

                # Bias (H1)
                self.b_H1 -= self.lr_b * D_post_H1

                # Weight (I -- H1)
                for k1 in range(2):
                    for i in range(3):
                        for j in range(3):
                            self.W_ItoH1[i, j] -= self.lr * np.sum(image[i:i + 15:2, j:j + 15:2] * D_pre_H1[k1])

            train_acc /= len(train_images)
            train_acc *= 100
            train_loss /= len(train_images)
            print('(Train) Accuracy : {:.1f}%, Loss : {:.5f}'.format(train_acc, train_loss), end='   ')
            self.train_loss_list.append(train_loss)
            self.train_acc_list.append(train_acc)

            # Validation
            valid_acc = 0
            valid_loss = 0
            for valid_idx in range(len(valid_images)):
                image = valid_images[valid_idx]
                label = valid_labels[valid_idx]

                softmax_class = HierarchicalNet.softmax(self.feedforward(image))
                if np.argmax(softmax_class) == np.argmax(label):
                    valid_acc += 1
                valid_loss += HierarchicalNet.softmax_cross_entropy(label, softmax_class)
            valid_acc /= len(valid_images)
            valid_acc *= 100
            valid_loss /= len(valid_images)
            print('(Valid) Accuracy : {:.1f}%, Loss : {:.5f}'.format(valid_acc, valid_loss))
            self.valid_loss_list.append(valid_loss)
            self.valid_acc_list.append(valid_acc)

    def test(self, test_dataset):
        test_images, test_labels = test_dataset[0], test_dataset[1]
        test_acc = 0
        test_loss = 0
        for test_idx in range(len(test_images)):
            image, label = test_images[test_idx], test_labels[test_idx]

            softmax_class = HierarchicalNet.softmax(self.feedforward(image))
            if np.argmax(softmax_class) == np.argmax(label):
                test_acc += 1
            test_loss += HierarchicalNet.softmax_cross_entropy(label, softmax_class)

        test_acc /= len(test_images)
        test_loss /= len(test_images)
        print('                    (Test) Accuracy : {:.4f}, Loss : {:.5f}'.format(test_acc, test_loss))

    def visualize(self):
        epochs = np.array(self.epoch_list)
        train_losses = np.array(self.train_loss_list)
        valid_losses = np.array(self.valid_loss_list)

        train_acces = np.array(self.train_acc_list)
        valid_acces = np.array(self.valid_acc_list)

        plt.figure(0)
        plt.plot(epochs, train_losses, 'r-', label='Train loss')
        plt.plot(epochs, valid_losses, 'b:', label='Valid loss')
        plt.title('Train/Validation Loss')
        plt.legend()

        plt.figure(1)
        plt.plot(epochs, train_acces, 'r-', label='Train acc')
        plt.plot(epochs, valid_acces, 'b:', label='Valid acc')
        plt.title('Train/Validation Acc')
        plt.legend()

        plt.show()

if __name__ == '__main__':
    hn1 = HierarchicalNet(epochs=500)
    hn1.exec_all('../digit data', SOBEL_X)
