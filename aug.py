from PIL import Image
import numpy as np

temp = 0
for k in range(10):
    for label in range(10):
        fn = './digit data/{}.{}.png'.format(label, k)
        img = Image.open(fn)
        img = np.array(img)
        n = -1
        new_k = 10 + temp * 4
        for i in range(4):
            n = np.random.randint(1, 6)
            result_img = img
            if n == 1:
                result_img = np.fliplr(img)
            elif n == 2:
                for i in range(16, 1, -1):
                    for j in range(16):
                        if i < 14:
                            result_img[j, i] = img[j, i - 2]
                        elif i < 15:
                            result_img[j, i] = 255
            elif n == 3:
                for j in range(16):
                    for i in range(16):
                        if i < 14:
                            result_img[j, i] = img[j, i + 2]
            elif n == 4:
                for j in range(16):
                    for i in range(16):
                        if j < 14:
                            result_img[j, i] = img[j + 2, i]
                        else:
                            img[j, i] = 255
            elif n == 5:
                for j in range(15, -1, -1):
                    for i in range(16):
                        if j >= 2:
                            result_img[j, i] = img[j - 2, i]
                        else:
                            result_img[j, i] = 255
            elif n == 6:
                noise = np.random.randint(0, 127, size=(16, 16, 3), dtype='uint8')
                for i in range(16):
                    for j in range(16):
                        for k in range(3):
                            if img[i, j, k] < 127:
                                result_img[i, j, k] += noise[i, j, k]

            img = Image.fromarray(result_img)
            img.save('./aug digit data/{}.{}.png'.format(label, new_k))
            img = np.array(img)
            new_k += 1
    temp += 1